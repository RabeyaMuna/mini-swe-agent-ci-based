"""
Template-guided instructions - like mini-swe-agent.

Instead of asking model "what do you want to do?",
we TELL model "do this specific thing now".

This matches mini-swe-agent's template approach that works with weak models.
"""


def build_simple_instruction(
    problem_context: str,
    files_read: list[str],
    files_written: list[str],
    previous_result: str = '',
) -> str:
    """
    Template-guided workflow like mini-swe-agent.

    Agent controls workflow, model just generates content.
    This works with weak models like GLM-5.2.

    Workflow:
    Step 1: COMMAND model to read files
    Step 2: COMMAND model to generate fixed content
    Step 3: Agent auto-writes the content (model doesn't decide)
    Step 4: COMMAND model to mark done
    """

    # STEP 1: No files read yet - COMMAND to read
    if not files_read:
        return f"""{problem_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMAND: READ THE FILE TO SEE THE ERROR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Look at the problem above. Find the file path mentioned.

YOUR REQUIRED OUTPUT (copy this format exactly):
{{"tool": "read_file", "args": {{"file_path": "EXACT_PATH_FROM_PROBLEM"}}}}

Example if problem mentions "error in test.py":
{{"tool": "read_file", "args": {{"file_path": "path/to/test.py"}}}}

OUTPUT ONLY THE JSON - NOTHING ELSE
"""

    elif files_read and not files_written:
        # STEP 2: COMMAND to write - like mini-swe-agent templates
        files_list = '\n'.join(f'  ✓ {f}' for f in files_read[:5])

        # Extract the file path from the first file read
        file_to_fix = files_read[0] if files_read else 'file.py'

        return f"""{problem_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMAND: WRITE THE COMPLETE FIXED FILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You read: {file_to_fix}

Previous result:
{previous_result}

Now you MUST write the COMPLETE corrected version.

YOUR REQUIRED OUTPUT (use this exact format):
{{"tool": "write_file", "args": {{"file_path": "{file_to_fix}", "content": "COMPLETE_FILE_WITH_ALL_FIXES_APPLIED"}}}}

CRITICAL RULES:
1. "content" must contain the ENTIRE file (all lines, not just changes)
2. Apply ALL fixes mentioned in the problem (type hints, imports, etc.)
3. Keep all other code unchanged

Example structure:
{{"tool": "write_file", "args": {{"file_path": "{file_to_fix}", "content": "# Complete file\\nimport typing\\n\\ndef function() -> None:\\n    pass\\n"}}}}

OUTPUT ONLY THE JSON WITH COMPLETE FILE CONTENT
"""

    elif files_written:
        # STEP 3: COMMAND to mark done
        files_list = '\n'.join(f' {f}' for f in files_written)

        return f"""{problem_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMAND: MARK TASK COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Files you fixed:
{files_list}

Your fixes are complete. Mark the task as done.

YOUR REQUIRED OUTPUT (exact format):
{{"tool": "done", "args": {{"notes": "Fixed errors in {len(files_written)} file(s)"}}}}

OUTPUT ONLY THIS JSON
"""

    # Fallback - shouldn't reach here
    return """Continue working.

OUTPUT: {"tool": "read_file", "args": {"file_path": "file.py"}}
"""
