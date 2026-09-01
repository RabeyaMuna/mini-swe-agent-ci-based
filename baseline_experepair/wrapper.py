"""
ExpeRepair Simple Baseline - Matching Real ExpeRepair Flow

Follows ExpeRepair's exact approach:
1. System: "You are maintaining the project"
2. User: "<issue>CI failure description</issue>"
3. User: "Code at buggy locations" (from log_details.json)
4. User: WRITE_PATCH_PROMPT (2-phase: analysis + implementation)

NO DIFF - only issue + code locations
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

import litellm
from utilities.model_token_config import get_output_max_tokens


# Real ExpeRepair system prompt
SYSTEM_PROMPT = """You are a software developer maintaining a GitHub project.
You are working on a CI failure issue.
Your task is to write a patch that resolves this issue."""

# Real ExpeRepair 2-phase prompt (from agent_write_patch.py)
WRITE_PATCH_PROMPT = """
### Phase 1: FIX ANALYSIS
1. Review the issue description and state clearly what the problem is.
2. Analyze the provided code context and specify where the problem occurs in the code.
3. State clearly the best practices to take into account in the fix.
4. State clearly how to fix the problem.

### Phase 2: FIX IMPLEMENTATION
1. Focus on making minimal, precise, and relevant changes to resolve the issue.
2. Include any necessary imports introduced by the patch.
3. Write the patch using the strict format specified below:
- Each modification must be enclosed in:
  - `<file>...</file>`: replace `...` with actual file path.
  - `<original>...</original>`: replace `...` with the original code snippet from the provided code locations.
  - `<patched>...</patched>`: replace `...` with the fixed version of the original code.
- The `<original>` block must contain an EXACT, continuous block of code from the provided code locations, as the system relies on this for locating modifications.
- When adding original code and patched code, pay attention to indentation, as the code is in Python.
- DO NOT include line numbers in the patch.
- You can write up to three modifications if needed.

EXAMPLE PATCH FORMAT:
# modification 1
```
<file>path/to/file.py</file>
<original>
exact code from file
</original>
<patched>
fixed code
</patched>
```

# modification 2
```
<file>path/to/file2.py</file>
<original>
exact code from file2
</original>
<patched>
fixed code
</patched>
```
"""


def generate_patch_experepair_baseline(
    issue_description: str,
    changed_files: List[str],
    repo_path: str,
    model: str = "deepseek-v4-flash",
    diff: str = "",
    workflow: str = "",
    validation_commands: str = "",
    memory_context: Optional[Dict] = None,
    sha_fail: str = "",
    instance_id: str = ""
) -> Dict:
    """
    ExpeRepair baseline matching REAL ExpeRepair flow.

    Flow (exactly like ExpeRepair):
    1. Load error from log_details.json
    2. Get decomposed problems from memory_context (MemoryPlugin)
    3. Build code context at buggy locations
    4. Call LLM with: issue + code + 2-phase prompt
    5. Extract patch
    """
    # 1. Load error analysis from log_details.json
    lookup = sha_fail or instance_id
    error_context = []
    failure_signals = []
    relevant_files = changed_files
    error_types = []

    try:
        with open("data/log_details.json") as f:
            log_cache = json.load(f)
        for entry in log_cache:
            if entry.get('sha_fail') == lookup or entry.get('id') == lookup:
                error_context = entry.get('error_context', [])
                failure_signals = entry.get('failure_signals', [])

                # Get relevant files
                rf_list = entry.get('relevant_files', [])
                if rf_list and isinstance(rf_list[0], dict):
                    relevant_files = [rf['file'] for rf in rf_list if rf.get('file')]
                    if not relevant_files:
                        relevant_files = changed_files

                # Get error types
                et_list = entry.get('error_types', [])
                if et_list and isinstance(et_list[0], dict):
                    error_types = [et.get('category', '') for et in et_list]
                break
    except Exception as e:
        print(f"Warning: Could not load log_details: {e}")
        error_context = [issue_description[:1000]]

    # 2. Build issue description (like ExpeRepair's <issue>)
    issue_text = "<issue>\n"
    issue_text += "## CI Failure\n\n"

    if error_context:
        issue_text += "### Error Description\n"
        issue_text += '\n'.join(error_context) + "\n\n"

    if failure_signals:
        issue_text += "### Failure Signals\n"
        for sig in failure_signals:
            issue_text += f"- {sig}\n"
        issue_text += "\n"

    if error_types:
        issue_text += f"### Error Type\n{', '.join(error_types)}\n\n"

    # Add decomposed problems from memory
    if memory_context and memory_context.get("problems"):
        issue_text += "### Decomposed Problems\n"
        for i, prob in enumerate(memory_context["problems"][:3], 1):
            issue_text += f"{i}. {prob.get('problem', '')}\n"
            if prob.get('root_cause'):
                issue_text += f"   - Root Cause: {prob.get('root_cause')}\n"
        issue_text += "\n"

    issue_text += "</issue>"

    # 3. Build code context (like ExpeRepair's buggy locations)
    code_context = "Below are the code locations that may be related to this CI failure:\n\n"

    for fpath in relevant_files[:5]:  # Top 5 files
        full_path = Path(repo_path) / fpath
        if not full_path.exists():
            continue

        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

            # Show file with line numbers (like ExpeRepair's BugLocation format)
            # Note: Line numbers are for reference only - LLM should NOT include them in <original> blocks
            code_context += f"**File: {fpath}**\n```python\n"
            for i, line in enumerate(lines[:150], 1):
                code_context += f"{i:4d}  {line}\n"
            code_context += "```\n\n"
        except Exception as e:
            print(f"Warning: Could not read {fpath}: {e}")

    code_context += "Note that you DO NOT NEED to modify every file; you should think what changes are necessary for resolving the issue, and only propose those modifications."

    # 4. Build messages (exactly like ExpeRepair)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": issue_text},
        {"role": "user", "content": code_context},
        {"role": "user", "content": WRITE_PATCH_PROMPT}
    ]

    # 5. Call LLM
    llm_model = model
    if model == "deepseek-v4-flash":
        llm_model = "openrouter/deepseek/deepseek-v4-flash"
    elif model.startswith("minimax"):
        llm_model = f"openrouter/{model}"
    elif not llm_model.startswith(("openrouter/", "gpt-")):
        llm_model = f"openrouter/{model}"

    # Determine max_tokens based on model capacity
    # Use mini-swe-agent's model configuration system
    # This handles all models (deepseek-v4-flash, minimax, glm, gpt, etc.)
    max_output = get_output_max_tokens(model)  # Pass original model key

    # Use 90% of max for safety (leave buffer for prompt/response overhead)
    max_tokens = int(max_output * 0.9)

    # Ensure reasonable bounds: at least 4096 for patch generation
    max_tokens = max(4096, max_tokens)

    print(f"Using max_tokens={max_tokens} (90% of {max_output}) for {model}")

    try:
        response = litellm.completion(
            model=llm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens
        )

        patch_raw = response.choices[0].message.content or ""

        # Check if response was truncated
        finish_reason = response.choices[0].finish_reason
        was_truncated = finish_reason == "length"

        # Save raw response for debugging
        print(f"\n{'='*60}")
        print(f"RAW LLM RESPONSE ({len(patch_raw)} chars):")
        if was_truncated:
            print(f"⚠️  WARNING: Response was TRUNCATED (finish_reason={finish_reason})")
            print(f"⚠️  The model hit max_tokens={max_tokens} limit!")
            print(f"⚠️  This likely means the response is incomplete.")
        print(f"{'='*60}")
        print(patch_raw)  # Full response
        print(f"{'='*60}\n")

        if was_truncated:
            print("⚠️  TRUNCATION DETECTED!")
            print("The LLM response was cut off before completing.")
            print("This usually means the response didn't reach Phase 2 (patch generation).")
            print("Possible solutions:")
            print("  1. Increase model's max_output_tokens")
            print("  2. Reduce code context (show fewer files/lines)")
            print("  3. Use a model with larger output capacity")
            print()

        # Convert ExpeRepair format to git diff (using their conversion logic)
        try:
            from baseline_experepair.experepair_core.agents.patch_utils import parse_edits, apply_edit
            from baseline_experepair.experepair_core import utils as apputils
            import subprocess

            # Parse <file><original><patched> blocks
            edits = parse_edits(patch_raw)
            print(f"Parsed {len(edits)} edits from LLM response")

            # Check if LLM incorrectly included line numbers (common mistake)
            for i, edit in enumerate(edits):
                first_line = edit.before.split('\n')[0] if edit.before else ''
                # Check if first line starts with digits followed by whitespace (line number format)
                import re
                if re.match(r'^\s*\d+\s+', first_line):
                    print(f"⚠️  Warning: Edit {i+1} may include line numbers in <original>!")
                    print(f"   First line: {first_line[:50]}")
                    print(f"   This will cause matching to fail!")

            if not edits:
                print("⚠️  Warning: No edits parsed from LLM response!")
                print("This could mean:")
                print("  1. LLM didn't generate <file><original><patched> format")
                print("  2. LLM said no changes needed")
                print("  3. Response format was incorrect")
                print()
                print("DEBUGGING: Checking for format markers...")
                has_file = '<file>' in patch_raw
                has_original = '<original>' in patch_raw
                has_patched = '<patched>' in patch_raw
                has_code_fence = '```' in patch_raw
                print(f"  - Has <file> tag: {has_file}")
                print(f"  - Has <original> tag: {has_original}")
                print(f"  - Has <patched> tag: {has_patched}")
                print(f"  - Has ``` code fence: {has_code_fence}")
                print()
                if not has_file:
                    print("  → LLM did not generate <file> tags!")
                    print("  → Likely just provided analysis or said 'no fix needed'")
                elif not has_code_fence:
                    print("  → LLM generated tags but no code fence (```)!")
                    print("  → Parser requires edits to be in ``` blocks")
                git_diff = ""
            else:
                # Apply edits to actual files and generate git diff
                with apputils.cd(repo_path):
                    # Clean any existing changes first
                    apputils.repo_clean_changes()

                    # Apply each edit
                    applied_count = 0
                    for i, edit in enumerate(edits, 1):
                        target_file = edit.filename
                        print(f"\nEdit {i}/{len(edits)}: {target_file}")

                        # Find file (model may use short name)
                        found_file = apputils.find_file(repo_path, target_file)
                        if not found_file:
                            print(f"  ✗ File not found: {target_file}")
                            continue

                        print(f"  ✓ Found file: {found_file}")

                        # Debug: show what we're trying to match
                        print(f"    Trying to match {len(edit.before.split(chr(10)))} lines from <original>")

                        result = apply_edit(edit, found_file)
                        if result:
                            applied_count += 1
                            print(f"  ✓ Edit applied successfully")
                        else:
                            print(f"  ✗ Edit could not be matched in file")
                            print(f"    <original> content (first 200 chars):")
                            print(f"    {edit.before[:200]}")

                            # Try to help debug why it didn't match
                            try:
                                with open(found_file, 'r') as f:
                                    file_content = f.read()
                                    if edit.before[:50] in file_content:
                                        print(f"    → Found partial match in file")
                                    else:
                                        print(f"    → No partial match found")
                                        # Show first few lines of actual file
                                        print(f"    Actual file content (first 200 chars):")
                                        print(f"    {file_content[:200]}")
                            except Exception as debug_err:
                                print(f"    Debug error: {debug_err}")

                    if applied_count == 0:
                        print(f"\n{'='*60}")
                        print(f"⚠️  DIAGNOSIS: No edits could be applied (0/{len(edits)})")
                        print(f"{'='*60}")
                        print("ROOT CAUSES:")
                        print("  1. Code in <original> doesn't match actual file content")
                        print("     → LLM may have hallucinated or used outdated code")
                        print("     → File may have been modified since context was generated")
                        print("     → LLM may have included line numbers in <original>")
                        print("  2. Files don't exist in repo at expected paths")
                        print("  3. Whitespace/indentation mismatches")
                        print()
                        print("RESULT: Empty patch (applicable=False)")
                        print(f"{'='*60}\n")
                        git_diff = ""
                    else:
                        # Generate git diff
                        diff_cmd = ['git', 'diff', '--no-color', '--no-ext-diff']
                        result = subprocess.run(
                            diff_cmd,
                            capture_output=True,
                            text=True,
                            cwd=repo_path
                        )
                        git_diff = result.stdout

                        # Clean changes after getting diff
                        apputils.repo_clean_changes()

                        print(f"Applied {applied_count}/{len(edits)} edits, diff size: {len(git_diff)} chars")

        except Exception as e:
            print(f"Warning: Failed to convert to git diff: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return raw format
            git_diff = patch_raw.strip()

        # Validate: Can the patch be applied?
        applicable = False
        validation_error = ""

        if git_diff:
            try:
                # Test if patch can be applied (using ExpeRepair's method)
                from tempfile import NamedTemporaryFile

                with apputils.cd(repo_path):
                    # Clean first
                    apputils.repo_clean_changes()

                    # Try to apply the patch
                    with NamedTemporaryFile(buffering=0, suffix=".diff", delete=False) as f:
                        f.write(git_diff.encode())
                        temp_patch = f.name

                    try:
                        apply_cmd = ["git", "apply", "--check", temp_patch]
                        result = subprocess.run(
                            apply_cmd,
                            capture_output=True,
                            text=True,
                            cwd=repo_path
                        )

                        if result.returncode == 0:
                            applicable = True
                            print("✓ Patch is applicable!")
                        else:
                            applicable = False
                            validation_error = result.stderr
                            print(f"✗ Patch not applicable: {result.stderr[:200]}")
                    finally:
                        import os
                        os.unlink(temp_patch)
                        # Clean any changes
                        apputils.repo_clean_changes()

            except Exception as e:
                validation_error = str(e)
                print(f"Validation error: {e}")

        # Calculate cost
        cost = 0.0
        if hasattr(response, 'usage') and response.usage:
            cost = (response.usage.prompt_tokens + response.usage.completion_tokens) / 1000 * 0.001

        return {
            "patch": git_diff.strip(),
            "cost": cost,
            "model": model,
            "method": "experepair_validated",
            "applicable": applicable,
            "validation_error": validation_error if not applicable else ""
        }

    except Exception as e:
        print(f"LLM Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "patch": "",
            "cost": 0.0,
            "model": model,
            "method": "experepair_exact",
            "error": str(e)
        }
