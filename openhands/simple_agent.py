"""
Simple template-based agent like mini-swe-agent.

No complex phases, no dynamic instructions.
Just: template → model → bash command → result → repeat
"""

import re
import subprocess
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

load_dotenv()


SYSTEM_TEMPLATE = """You are a helpful assistant that fixes code issues.

Your response must contain exactly ONE bash code block with ONE command.
Include a THOUGHT section before your command explaining your reasoning.

Format:
THOUGHT: Your analysis here

```bash
your_command_here
```

Rules:
1. Output exactly one bash command per response
2. Use bash commands to read files, edit files, run tests
3. When done, output: echo COMPLETE_TASK
"""


INSTANCE_TEMPLATE = """Solve this CI failure:

{problem_description}

{fault_localization}

Repository: {repository}
Commit: {commit}
Current directory: {work_dir}

INSTRUCTIONS:
1. Read the files listed above that have errors
2. Fix the errors using sed or heredoc
3. When all fixes are done: echo COMPLETE_TASK

IMPORTANT:
- Use `sed -i ''` on Mac OS X (not `sed -i`)
- Fix ALL occurrences, not just one (use /g flag in sed)
- Do NOT run mypy or tests - just fix the code
- Output exactly ONE bash command per response
"""


OBSERVATION_TEMPLATE = """<returncode>{returncode}</returncode>
<output>
{output}
</output>"""


def extract_bash_from_response(response: str) -> str | None:
    """Extract bash command from model response."""
    # Try ```bash
    match = re.search(r'```bash\s*\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ```
    match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def extract_fault_localization_simple(problem_text: str) -> str:
    """Extract fault localization from problem description."""
    import re

    faults = []

    # Pattern 1: Python errors - File "path.py", line 42
    for match in re.finditer(r'File "([^"]+)", line (\d+)', problem_text):
        faults.append(f'  - {match.group(1)}:{match.group(2)}')

    # Pattern 2: mypy errors - path.py:42: error: message
    for match in re.finditer(
        r'([a-zA-Z0-9_./\-]+\.py):(\d+):\s*error:\s*([^\n]+)', problem_text
    ):
        faults.append(f'  - {match.group(1)}:{match.group(2)} - {match.group(3)[:80]}')

    # Pattern 3: pytest failures
    for match in re.finditer(
        r'([a-zA-Z0-9_./\-]+\.py)::([\w_]+)\s+FAILED', problem_text
    ):
        faults.append(f'  - {match.group(1)} - test {match.group(2)} failed')

    if faults:
        return 'FAULT LOCALIZATION - Files with errors:\n' + '\n'.join(faults[:5])
    else:
        return 'FAULT LOCALIZATION: Analyze the problem description to identify files to fix.'


def run_simple_agent(
    problem_description: str,
    repository: str,
    commit: str,
    work_dir: Path,
    model: str,
    max_steps: int = 30,
) -> dict[str, Any]:
    """
    Run simple template-based agent (like mini-swe-agent).

    Returns:
        {"patch": str, "cost": float, "steps": int}
    """
    messages = []
    total_cost = 0.0
    step = 0

    # Extract fault localization
    fault_loc = extract_fault_localization_simple(problem_description)

    # Initial messages
    system_msg = SYSTEM_TEMPLATE
    instance_msg = INSTANCE_TEMPLATE.format(
        problem_description=problem_description,
        fault_localization=fault_loc,
        repository=repository,
        commit=commit,
        work_dir=str(work_dir),
    )

    messages.append({'role': 'system', 'content': system_msg})
    messages.append({'role': 'user', 'content': instance_msg})

    while step < max_steps:
        step += 1

        # Query model
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=4000,
            )
            total_cost += (response.usage.total_tokens / 1000) * 0.01
        except Exception as e:
            print(f'Model error: {e}')
            break

        response_text = response.choices[0].message.content or ''
        messages.append({'role': 'assistant', 'content': response_text})

        print(f'\nStep {step}: {response_text[:100]}...')

        # Extract bash command
        bash_command = extract_bash_from_response(response_text)
        if not bash_command:
            print('No bash command found')
            continue

        print(f'  Command: {bash_command[:80]}')

        # Check for completion
        if 'COMPLETE_TASK' in bash_command.upper():
            print('Task marked complete')
            break

        # Execute command
        try:
            # Fix sed -i for Mac
            import platform

            if (
                platform.system() == 'Darwin'
                and 'sed -i' in bash_command
                and "sed -i ''" not in bash_command
            ):
                bash_command = bash_command.replace('sed -i ', "sed -i '' ")

            result = subprocess.run(
                bash_command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = result.stdout + result.stderr
            if len(output) > 5000:
                output = (
                    output[:2500] + '\n... (output truncated) ...\n' + output[-2500:]
                )

            observation = OBSERVATION_TEMPLATE.format(
                returncode=result.returncode,
                output=output or '(no output)',
            )

        except Exception as e:
            observation = f'<error>{str(e)}</error>'

        print(
            f'  Return code: {result.returncode if "result" in locals() else "error"}'
        )

        # Add observation
        messages.append({'role': 'user', 'content': observation})

    # Generate patch
    try:
        diff_result = subprocess.run(
            ['git', 'diff', '--binary', commit],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        patch = diff_result.stdout
    except Exception:
        patch = ''

    return {
        'patch': patch,
        'cost': total_cost,
        'steps': step,
    }
