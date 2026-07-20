"""
Bash-based instructions for OpenHands - works with ANY model.

Instead of JSON tools, models output bash commands like mini-swe-agent.
This works for weak models (GLM-5.2) and strong models (GPT-4) alike.

DYNAMIC: Handles 567+ different CI failure patterns flexibly.
"""


def _get_phase_instruction(
    files_read: list[str], files_written: list[str], faulty_files: list[dict]
) -> str:
    """Determine what phase we're in and return appropriate instruction."""

    # Phase 1: Haven't read any files yet - need to read faulty files
    if not files_read and faulty_files:
        loc = faulty_files[0]
        file_to_read = loc['file_path']
        error_msg = loc.get('error', '')
        line_num = loc.get('line_number')

        instruction = f"""STEP 1: Read the file that has the error.

FILE TO FIX: {file_to_read}"""

        if line_num:
            instruction += f'\nLINE: {line_num}'

        if error_msg:
            instruction += f'\nERROR: {error_msg}'

        instruction += f"""

Read this file to see the current code:
```bash
cat {file_to_read}
```"""
        return instruction

    # Phase 2: Read files but haven't written fix yet - need to write fix
    if files_read and not files_written and faulty_files:
        file_to_fix = files_read[0]
        loc = faulty_files[0]
        error_msg = loc.get('error', '')

        instruction = f"""STEP 2: Write the fix.

FILE: {file_to_fix}"""

        if error_msg:
            instruction += f'\nERROR TO FIX: {error_msg}'

        instruction += """

Based on the error above and the file content you read, write the fix.

IMPORTANT: Fix ALL occurrences in the file, not just one.
- Use /g flag in sed for global replacement
- Or use pattern matching to fix all similar cases

Examples:
- Fix all functions: sed -i 's/def \\([a-zA-Z_][a-zA-Z0-9_]*\\)():/def \\1() -> None:/g' file
- Fix specific pattern: sed -i 's/old_pattern/new_pattern/g' file
- Full rewrite: cat > file << 'EOF' ... EOF

Output the fix command now."""
        return instruction

    # Phase 3: Have written fixes - mark complete
    if files_written:
        return """STEP 3: Mark complete.

Files have been fixed. Complete the task:
```bash
echo COMPLETE_TASK
```"""

    # Default: read first faulty file
    if faulty_files:
        return f'Read the file: {faulty_files[0]["file_path"]}'

    return 'Analyze the problem and read the file that needs fixing.'


def build_bash_instruction(
    problem_context: str,
    files_read: list[str],
    files_written: list[str],
    step_count: int = 0,
    previous_result: str = '',
    faulty_files: list[dict] = None,
) -> str:
    """
    Build bash-based instruction for any model.

    Models output bash commands in markdown blocks.
    This is familiar from training and works universally.
    """

    # Show fault localization results (first step only)
    fault_loc_section = ''
    if step_count == 0 and faulty_files:
        fault_loc_section = '\nFAULT LOCALIZATION (files to fix):\n'
        for i, loc in enumerate(faulty_files[:3], 1):
            file_path = loc['file_path']
            line_num = loc.get('line_number')
            error = loc.get('error', '')
            if line_num:
                fault_loc_section += f'{i}. {file_path}:{line_num}'
            else:
                fault_loc_section += f'{i}. {file_path}'
            if error:
                fault_loc_section += f' - {error[:80]}'
            fault_loc_section += '\n'

    # Show progress and detect if stuck in read-only loop
    progress = ''
    stuck_in_reading = files_read and not files_written and len(files_read) >= 2

    if files_read:
        progress += f'\nFiles explored: {len(files_read)}'
        if stuck_in_reading:
            progress += ' WARNING - TOO MANY READS, MUST WRITE NOW'
    if files_written:
        progress += f'\nFiles modified: {len(files_written)}'
        progress += '\n' + '\n'.join(f'  {f}' for f in files_written)

    # Show previous result if available
    result_section = ''
    if previous_result and step_count > 0:
        result_section = f'\nPrevious command result:\n{previous_result}\n'

    # ANTI-LOOP: Force write if stuck in read phase
    force_write_warning = ''
    if stuck_in_reading:
        force_write_warning = f"""
WARNING: YOU HAVE READ {len(files_read)} FILES BUT WRITTEN 0 FIXES.
DO NOT READ MORE. YOU MUST WRITE THE FIX NOW.
OUTPUT A WRITE COMMAND (sed -i OR cat >) IN YOUR NEXT RESPONSE.
"""

    # Build instruction
    return f"""{problem_context}
{fault_loc_section}

{'=' * 70}
INSTRUCTIONS: Use Bash Commands to Fix the Problem
{'=' * 70}

{progress}
{result_section}
{force_write_warning}

WORKFLOW:
{_get_phase_instruction(files_read, files_written, faulty_files)}

BASH COMMAND RULES:
- Output commands in ```bash code blocks
- One command per response

Command patterns:
- Read file: cat path/to/file.py
- Fix (in-place): sed -i 's/old/new/g' file.py
- Fix (full file): cat > file.py << 'EOF' ... EOF
- Complete: echo COMPLETE_TASK

Repository location: {problem_context.split('Repository:')[1].split()[0] if 'Repository:' in problem_context else '.'}

EXAMPLES:

Example 1 - Find and read files:
```bash
find . -name "*test*.py" -type f | head -10
```

Example 2 - Read a specific file:
```bash
cat framework/py/flwr/common/inflatable_test.py
```

Example 3 - Write fixed file (COMPLETE content):
```bash
cat > framework/py/flwr/common/inflatable_test.py << 'EOF'
# Copyright header here
import typing

def test_function() -> None:
    # Fixed with type annotation
    pass
EOF
```

Example 4 - Mark complete:
```bash
echo COMPLETE_TASK
```

OUTPUT YOUR BASH COMMAND NOW (in ```bash block):
"""
