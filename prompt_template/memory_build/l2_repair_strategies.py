"""
L2 Repair Strategies Prompts.

Prompts for generating repair strategies from L1 problems.
"""

from typing import Dict, List
import json


def build_l2_prompt(
    l1_memory: Dict,
    automated_tools: List[Dict],
    sampling_info: Dict = None,
) -> str:
    """
    Build L2 prompt to generate repair strategies from L1 problems.

    LEAN VERSION: Trust the LLM to analyze problems and match to automation tools.
    No hardcoded templates, no repetitive examples, fully dynamic.

    Args:
        l1_memory: L1 memory with problems (may be sampled)
        automated_tools: List of available automation tools (self-documenting)
        sampling_info: Optional sampling metadata (if L1 was sampled)

    Returns:
        Prompt string for LLM
    """
    issue_id = l1_memory.get("issue_id")
    repo = l1_memory.get("repo")
    workflow = l1_memory.get("workflow")
    problems = l1_memory.get("problems", [])
    changed_files = l1_memory.get("changed_files", [])

    # Sampling note
    sampling_note = ""
    if sampling_info and sampling_info.get("was_sampled"):
        sampling_note = f"""
NOTE: L1 sampled (~{sampling_info.get("total_size", 0):,} tokens). Problems with >10 files show first 5 + files_metadata with total count.
"""

    prompt = f"""Analyze CI failure and generate repair strategies (L2).

ISSUE: {issue_id} | REPO: {repo} | WORKFLOW: {workflow}
{sampling_note}

=== CHANGED FILES ===
{json.dumps(changed_files, indent=2)}

=== L1 PROBLEMS ({len(problems)} total) ===
{json.dumps(problems, indent=2)}

L1 Problem Schema:
- problem_id, verification_cmd, failure_type, problem, root_cause, fix_strategy
- files: affected paths (or files_metadata if >10 files: total_count, pattern, note)
- enabled: [problem_ids] revealed after fixing this (sequential dependencies)

=== AUTOMATION TOOLS ===
{json.dumps(automated_tools, indent=2)}

Each tool specifies:
- purpose: what it fixes
- fixes: list of problem types it addresses
- file_pattern: what files it works on
- install_command: how to install
- fix_command: how to run ({{{{file_or_dir}}}} = placeholder for target path)

=== YOUR TASK ===

1. **Analyze Problems**: For each L1 problem:
   - Check if problem.failure_type matches any tool.fixes array
   - Check if problem.files match any tool.file_pattern
   - Count files: use files_metadata.total_count if exists, else len(files)
   - Decision: IF (match exists AND files ≥ 10) → automation; ELSE → manual

2. **Group Problems**: Create strategies by grouping:
   - Problems with same root_cause → ONE strategy
   - Problems where A.enabled contains [B, C] → ONE strategy (A enables B and C)
   - Independent problems → SEPARATE strategies

3. **Generate Strategies**: For each strategy, build key_actions:
   - IF automation: Use tool.install_command, then tool.fix_command
     * Replace {{{{file_or_dir}}}} with: common parent directory if multiple files share one, else individual file
     * EFFICIENT: 20 files in src/utils/ → run on src/utils/ (NOT file-by-file)
   - IF manual: Use steps from L1.fix_strategy
   - ALWAYS end with: L1.verification_cmd to verify

4. **Output**: Return JSON with:
   - failure_identify: ["failure_type (validator) - N problems"] (use files_metadata.total_count when present)
   - repair_strategies: Array of strategy objects (see schema below)

=== ANALYSIS GUIDELINES ===

**Grouping Logic**:
- problem.enabled = [X, Y] means fixing this problem reveals problems X and Y
- Group enabled problems in ONE strategy, explain causal chain
- Share root_cause → group together; different causes → separate

**Automation Matching**:
- Match problem.failure_type to tool.fixes array (each tool lists what it fixes)
- Match problem.files extensions to tool.file_pattern
- Use tool.install_command and tool.fix_command from matched tool

**Signals** (observable error patterns):
- Extract actual error messages from L1.problem field
- Include file paths/patterns, command failures, tool versions
- Be specific: "mypy: error [arg-type] at helpers.py:42" not "type error"

**Key Actions**:
- Automation: tool.install_command → tool.fix_command with target → L1.verification_cmd
  * If multiple files in SAME directory: target = common parent directory (more efficient)
  * If files scattered across directories: target = repository root or common ancestor
  * Example: 20 files in src/utils/ → run tool on src/utils/ NOT file-by-file
- Manual: steps from L1.fix_strategy → L1.verification_cmd
- Config changes: specify exact file, section, key=value

CRITICAL OUTPUT FORMAT:

Return ONLY valid JSON with this structure:

IMPORTANT JSON FORMATTING:
- Your FIRST character MUST be {{ - NO text or backticks before
- Your LAST character MUST be }} - NO text or backticks after
- Do NOT wrap in backticks: NO ``` or ```json or ` - NONE AT ALL
- Do NOT add markdown, explanations, or any text outside the JSON
- The response will be passed DIRECTLY to json.loads() - it MUST parse perfectly

{{
  "issue_id": "{issue_id}",
  "repo": "{repo}",
  "workflow": "{workflow}",
  "total_problems": {len(problems)},
  "failure_identify": [
    "failure_type (validator) - N problems/files"
  ],
  "repair_strategies": [
    {{
      "step": 1,
      "failure_type": "<extract from L1 problem.failure_type>",
      "validation_cmd": "<extract from L1 problem.verification_cmd>",
      "applies_to_failures": ["<from failure_identify above>"],

      // USE PATTERN: "<failure_type> in <location> (<specific_error>)" for each problem
      "causal_chain": "<Describe relationship between problems using the pattern. Extract location from L1 files/files_metadata, error details from L1 problem field>",

      "summary": "<One-line description - what files/areas affected>",
      "intent": "<What this strategy achieves - be specific to YOUR issue>",

      // Reference problems using pattern, extract details from YOUR L1 data
      "reasoning": "<Why grouping these problems - use actual file names and error details from L1>",
      "rationale": "<Why this approach works - manual vs automation based on YOUR L1 error types>",

      "when_to_apply": "<Specific conditions from YOUR L1 - validators, error patterns, file patterns>",

      // OBSERVABLE PATTERNS that detect this failure (extract from L1 but format as actual CI log patterns)
      // CRITICAL: Include config/version info when relevant!
      "signals": [
        "<tool>: <actual_error_message_as_it_appears> at <file>:<line>",
        "<N> files matching <pattern> affected",
        "Config: <specific_version_constraint_or_setting> in <config_file>",
        "Command '<actual_command>' exits with code <exit_code>"
      ],

      // EXAMPLES of properly formatted signals (NOTICE CONFIG INFO):
      // Type error: "mypy: error: Incompatible type [arg-type] at helpers.py:42"
      // Format error: "67 files matching docs/source/**/*.rst with heading adornment errors"
      // Test error: "pytest: AssertionError: assert 200 == 401 in test_auth.py::test_login"
      // Import error: "ImportError: No module named 'numpy.typing' in utils/config.py"
      // Dependency: "poetry: Dependency conflict - typer requires click<8.2.0 but pyproject.toml has no constraint"
      // Version: "numpy 2.0 removed DTypeLike - need to update type annotations"
      // Command: "Command 'python -m mypy .' exits with code 1"

      "key_actions": [
        // CRITICAL RULES FOR STEPS:
        // 1. CONFIG CHANGES FIRST, CODE CHANGES SECOND
        // 2. For AUTOMATED fixes: exact command + "no manual check unless this fails"
        // 3. For MANUAL fixes: exact file, line, change needed
        // 4. Include specific config values (versions, constraints, settings)
        // 5. If any *.py file changes, add Ruff cleanup before final validation:
        //    "Install Ruff if missing: pip install ruff"
        //    "Run Ruff autofix: ruff check --fix <changed_python_file_or_dir>"
        //    "Run Ruff formatter: ruff format <changed_python_file_or_dir>"
        //
        // PATTERN A: Config + Code fix (e.g., dependency issues)
        //   "Step 1: CONFIG - Open <config_file> and modify <section>: set <key>=<value>"
        //   "Step 2: CODE - Open <source_file> and change <specific_code_location>: <exact_change>"
        //   "Step 3: Run Ruff cleanup if Python files changed: ruff check --fix <target> && ruff format <target>"
        //   "Step 4: Verify: <verification_cmd from L1> (should pass)"
        //
        // PATTERN B: Automated fix (formatting/linting, many files)
        //   "Step 1: Install <tool>: <exact install_command from AUTOMATED_TOOLS>"
        //   "Step 2: Run automated fix: <exact fix_command with file paths> (fixes automatically, no manual check unless this fails)"
        //   "Step 3: Run Ruff cleanup if Python files changed: ruff check --fix <target> && ruff format <target>"
        //   "Step 4: Verify: <verification_cmd> (should pass)"
        //
        // PATTERN C: Manual fix only (semantic changes, few files)
        //   "Step 1: Open <actual file from L1> at line <line_number if available>"
        //   "Step 2: Change <old_code> to <new_code> because <reason from L1 fix_strategy>"
        //   "Step 3: Run Ruff cleanup if Python files changed: ruff check --fix <target> && ruff format <target>"
        //   "Step 4: Verify: <verification_cmd from L1> (should pass)"
        //
        // EXAMPLES:
        // Config+Code: "Step 1: CONFIG - Open pyproject.toml [tool.poetry.dependencies] section: add click = '<8.2.0'"
        //              "Step 2: CODE - Open app.py line 42: change secondary=False to secondary=None"
        // Automated:   "Step 1: Install: pip install ruff"
        //              "Step 2: Run: ruff check --fix src/ (fixes automatically, no manual check unless this fails)"
        // Manual:      "Step 1: Open utils.py line 15"
        //              "Step 2: Change 'from numpy.typing import DTypeLike' to 'from typing import Any'"
      ],

      "pitfalls": [
        "<Common mistakes specific to THIS error type>",
        "<Tool-specific warnings if using automation>"
      ],

      "example_phrasing": "<Natural language using actual file names and error types from YOUR L1>"
    }}
  ]
}}

REMEMBER:
- Use files_metadata.total_count when available (not len(files))
- Choose automation tool only if appropriate (check file pattern + error type)
- Group problems by causal relationship, not just validation
- Include sequential dependencies from "enabled" field
- Make signals and key_actions SPECIFIC and EXECUTABLE

CRITICAL REQUIREMENTS:
- ALWAYS include "failure_type" and "validation_cmd" fields in each strategy
- SIGNALS must include config/version info when relevant (e.g., "click >= 8.2.0 constraint missing")
- KEY_ACTIONS must follow pattern:
  * CONFIG changes BEFORE code changes (if both needed)
  * For AUTOMATED fixes: exact install + run command + "no manual check unless this fails"
  * For MANUAL fixes: exact file, line number (if available), specific change
  * Include specific config values (versions, sections, keys)
- For AUTOMATION tools: Use exact commands from AUTOMATED_TOOLS list (install_command, fix_command)
- Separate config modification from code modification in steps
"""

    return prompt
