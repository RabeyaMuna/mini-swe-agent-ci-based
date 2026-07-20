# Simple Template-Based Agent

Rewrote OpenHands to match mini-swe-agent's simple template approach.

## What Changed

### Old Approach (Wrong)
- Complex dynamic phase detection
- Bash parser with multiple fallbacks
- Fault localization with step-by-step workflow
- Loop detection and anti-loop warnings
- 500+ lines of complex logic

Result: GLM-5.2 gets confused, loops forever, no patches generated

### New Approach (Simple)
- Fixed template (like mini-swe-agent)
- Model outputs ONE bash command per step
- Execute command, show result
- Repeat until done
- 150 lines total

## Template

```
SYSTEM: You are an assistant. Output exactly ONE bash command per response.

INSTANCE: Solve this CI failure: {problem}

Workflow:
1. Analyze the error
2. Read relevant files
3. Edit files to fix (use sed or heredoc)
4. When done: echo COMPLETE_TASK

OBSERVATION: <returncode>{code}</returncode><output>{output}</output>
```

## How It Works

1. Send system + instance templates to model
2. Model outputs: THOUGHT + ```bash command ```
3. Execute bash command
4. Format observation (returncode + output)
5. Add to messages
6. Repeat until COMPLETE_TASK or max steps

## Key Fixes

1. Mac OS X sed: Auto-convert `sed -i` to `sed -i ''`
2. Simple loop: No complex phase detection
3. Clear format: Model knows exactly what to output
4. No dynamic instructions: Template never changes

## Usage

```python
from simple_agent import run_simple_agent

result = run_simple_agent(
    problem_description=task['initial_message'],
    repository=task['repository'],
    commit=task['commit_sha'],
    work_dir=Path('/path/to/repo'),
    model='zai/glm-5.2',
    max_steps=30,
)

print(result['patch'])  # Git diff patch
print(result['cost'])   # Total cost
print(result['steps'])  # Steps taken
```

## Why This Works

Mini-swe-agent proved this template approach works with GLM-5.2. By copying their approach exactly:
- Fixed templates (not dynamic)
- ONE command per response
- Simple observation format
- No complex decision making

The model doesn't need to understand phases or workflows - it just follows the template pattern it was trained on.
