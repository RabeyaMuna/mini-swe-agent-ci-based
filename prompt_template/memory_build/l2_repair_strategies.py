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

    Args:
        l1_memory: L1 memory with problems (may be sampled)
        automated_tools: List of available automation tools
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
  NOTE: L1 data sampled for efficiency (~{sampling_info.get("total_size", 0):,} tokens)
- Problems with >10 files show first 5 + metadata
- Check 'files_metadata' for total count and pattern
"""

    prompt = f"""You are analyzing a CI failure to identify REPAIR STRATEGIES (L2).

ISSUE: {issue_id}
REPO: {repo}
WORKFLOW: {workflow}
{sampling_note}

=== CHANGED FILES ===
{json.dumps(changed_files, indent=2)}

=== L1 PROBLEMS ({len(problems)} total) ===
{json.dumps(problems, indent=2)}

Each L1 problem has:
- problem_id: Unique identifier
- verification_cmd: Which validator failed (e.g., "python -m mypy py")
- failure_type: Type of failure (e.g., "type checking", "formatting")
- problem: Error description
- root_cause: Why it occurred
- fix_strategy: How it was fixed
- files: Affected file paths (may be sampled)
- files_metadata: (if present) total_count, pattern, note
- enabled: Problems revealed after fixing this (sequential dependencies)

UNDERSTANDING SAMPLED DATA:
If files_metadata exists for a problem:
- files: Shows first 5 sample files
- files_metadata.total_count: ACTUAL total number of files (use this for counts)
- files_metadata.pattern: Common directory/file pattern
- Use total_count (not sample length) when deciding manual vs automation

=== DECISION TREE: HOW TO ANALYZE YOUR SPECIFIC ISSUE ===

FOR EACH L1 PROBLEM, FOLLOW THIS PROCESS:

Step 1: EXTRACT CORE INFO
→ failure_type = problem.failure_type
→ validator = extract tool name from problem.verification_cmd
→ error_detail = key phrase from problem.problem
→ location = problem.files[0] if single file, else files_metadata.pattern if exists, else "N files"

Step 2: DETERMINE FIX APPROACH
→ Check problem.failure_type:
  - "type checking" / "type error" / "mypy" → MANUAL (semantic)
  - "test failure" / "assertion" → MANUAL (logic)
  - "formatting" / "style" / "lint" → CHECK AUTOMATION
  - "import error" / "dependency" → MANUAL (code change)
  - "configuration" → MANUAL (config file)

→ If CHECK AUTOMATION:
  a. Analyze the PROBLEM FIRST (not just file extension):
     - Look at L1 failure_type and issue_type
     - Look at the validator/tool that detected it (from verification_cmd)
     - Look at the actual error message in L1 problem field

  b. Match PROBLEM to auto-fixable tools:
     - "format" or "lint" errors → Check if tool has auto-fix
       * ruff detected error → ruff check --fix (auto-fixes)
       * black detected error → black (auto-fixes)
       * isort detected error → isort (auto-fixes)
       * mdformat detected error → mdformat (auto-fixes)
       * docstrfmt detected error → docstrfmt (auto-fixes)

     - "type" errors (mypy, pyright) → MANUAL (no auto-fix)
     - "test" failures (pytest) → MANUAL (semantic issues)
     - "import" errors → MANUAL (need code understanding)
     - "dependency" conflicts → CONFIG change (not automation)

  c. Count files: use files_metadata.total_count if exists, else len(files)

  d. Final decision:
     - Tool CAN auto-fix + Files ≥ 10 + uniform error → USE AUTOMATION
     - Tool CANNOT auto-fix (mypy, pytest, etc.) → MANUAL
     - Files < 10 even if auto-fixable → Consider MANUAL (small enough)

  EXAMPLES:
  - L1 says "ruff: E501 line too long" in 50 files → USE ruff check --fix (automation)
  - L1 says "mypy: error: Incompatible types" in 3 files → MANUAL (no auto-fix)
  - L1 says "pytest: AssertionError" → MANUAL (semantic test failure)
  - L1 says "black would reformat" in 20 files → USE black (automation)

Step 3: BUILD PROBLEM REFERENCE
→ Pattern: "{{failure_type}} in {{location}} ({{error_detail}})"
→ Example construction:
  - failure_type = "type error"
  - location = "helpers.py" (from files[0])
  - error_detail = "Optional[Any] vs str" (extract from problem.problem)
  - Result: "Type error in helpers.py (Optional[Any] vs str)"

Step 4: GROUP PROBLEMS
→ Check problem.enabled field:
  - If problem A has "enabled": [problem_B] → GROUP A+B in ONE strategy
  - Explain: "Fixing A reveals B because..."
→ Check root_cause similarity:
  - Same root cause → GROUP in ONE strategy
  - Different root causes → SEPARATE strategies

=== AVAILABLE AUTOMATION TOOLS ===
{json.dumps(automated_tools, indent=2)}

HOW TO CHOOSE THE RIGHT AUTOMATION TOOL:

1. **Match file pattern AND error type**:
   - *.rst files + RST formatting errors (heading style, syntax) → docstrfmt
   - *.py files + docstring errors (PEP 257) → docformatter
   - *.py files + import errors → isort
   - *.py files + code formatting → black or ruff
   - *.md files + markdown formatting → mdformat
   - *.toml files + TOML formatting → taplo

2. **Check error type**:
   - Formatting/style/linting errors → likely has automation tool
   - Type checking/logic/semantic errors → NO automation, must be manual

3. **Check validator name**:
   - If validator = docstrfmt → use docstrfmt tool
   - If validator = black → use black tool
   - If validator = mypy/pylint/pytest → likely NO automation

4. **Consider file count and uniformity**:
   - 1-5 files → prefer manual fix (even if tool exists)
   - 10+ files with IDENTICAL error → automation strongly recommended
   - Error differs between files → manual fix required

CRITICAL DISTINCTIONS:
- **docstrfmt** = for *.rst documentation files (RST syntax)
- **docformatter** = for *.py docstrings (PEP 257 inside Python files)
- Example: "heading adornment style" in *.rst files → docstrfmt
- Example: "docstring not PEP 257" in *.py files → docformatter

CONFIG SPECIFICATION RULES:
When a fix requires config changes (dependencies, versions, settings), be SPECIFIC:

BAD (vague):
- "Add click version constraint"
- "Update dependency"
- "Fix version conflict"

GOOD (specific):
- "In pyproject.toml [tool.poetry.dependencies] section: add click = '<8.2.0'"
- "In requirements.txt: change numpy==1.24.0 to numpy>=2.0.0"
- "In .pre-commit-config.yaml: update mdformat-beautysh from rev: v0.3.0 to rev: v1.0.0"

INCLUDE in signals:
- Exact version constraints (e.g., "typer requires click<8.2.0")
- Config file names (e.g., "pyproject.toml missing constraint")
- Section names if applicable (e.g., "[tool.poetry.dependencies]")

INCLUDE in key_actions:
- Exact config file path
- Exact section/location
- Exact key and value to set
- Then code changes (if any)

=== TASK ===
Analyze these L1 problems and generate:

1. **failure_identify**: Summary of failure types found
   - Analyze L1 problems to identify distinct failure types
   - Extract validator name from verification_cmd (e.g., "mypy" from "python -m mypy py")
   - Group by failure_type and count problems OR files
   - IMPORTANT: If files_metadata exists, use total_count not sample size
   - Format: ["failure_type (validator) - N problems" or "- M files", ...]
   - Examples:
     * ["type_checking (mypy) - 3 problems"]
     * ["formatting (docstrfmt) - 67 files"] (use files_metadata.total_count)

2. **repair_strategies**: Actionable repair strategies (1-5 strategies)
   Each strategy must follow the schema with these REQUIRED fields:
   - step: Sequential step number (1, 2, 3...)
   - failure_type: Specific failure type being addressed (e.g., "type_checking", "formatting")
   - validation_cmd: The validator command (e.g., "mypy .", "ruff check .")
   - applies_to_failures: Which failure types from failure_identify this strategy addresses
   - causal_chain: Explain dependencies between problems and why they're related
   - summary: One-line description of the strategy
   - intent: What this strategy aims to achieve
   - reasoning: Why these L1 problems are grouped together
   - rationale: Why this approach works (technical explanation)
   - when_to_apply: Specific conditions when this strategy should be used
   - signals: Observable indicators that this strategy is needed (error messages, file patterns)
   - key_actions: Detailed step-by-step instructions (see AUTOMATION TOOL GUIDANCE below)
   - pitfalls: Common mistakes to avoid
   - example_phrasing: Natural language description for user

STRATEGY ORGANIZATION RULES:

1. **How to reference problems** (in causal_chain, reasoning, rationale):
   USE THIS PATTERN - adapt to YOUR specific issue:

   Pattern: "<failure_type> in <location> (<specific_error>)"

   WHERE to extract each part:
   - <failure_type>: From L1 problem.failure_type field
   - <location>:
     * Single file: Use actual filename from L1 problem.files[0]
     * Multiple files (<10): List filenames from L1 problem.files
     * Many files (10+): Use L1 problem.files_metadata.pattern if available, else first file + "and N others"
   - <specific_error>: Extract key error detail from L1 problem.problem field (the unique identifier)

   HOW to construct:
   1. Read L1 problem.failure_type → that's your failure_type
   2. Check L1 problem.files_metadata:
      - If exists and total_count > 10: Use files_metadata.pattern for location
      - Else: Use actual file path(s) from problem.files
   3. Extract the KEY error detail from problem.problem:
      - For type errors: the type mismatch (e.g., "Optional[Any] vs str")
      - For format errors: the format issue (e.g., "heading adornment style")
      - For import errors: what's missing (e.g., "module X not found")
      - For test errors: what assertion failed (e.g., "expected X got Y")

   ADAPT to your issue - these are just templates:
   - Type errors: "{{failure_type}} in {{file}} ({{type_A}} vs {{type_B}})"
   - Format errors: "{{failure_type}} in {{N}} {{pattern}} files ({{format_issue}})"
   - Import errors: "{{failure_type}} in {{file}} (missing {{module}})"
   - Test errors: "{{failure_type}} in {{test_file}} ({{assertion_detail}})"
   - Config errors: "{{failure_type}} in {{config_file}} ({{config_issue}})"

2. **Group by causal relationship**, not just validation:
   - If problems share root cause (e.g., config change triggered multiple validators) → ONE strategy
   - If problems are independent (different root causes) → SEPARATE strategies
   - If fixing one problem enables others (from "enabled" field) → SAME strategy

3. **Automation vs Manual**:
   - For mechanical/formatting errors with automation tool:
     * Specify exact automation tool and command
     * Include tool from available automation tools list
   - For semantic/logic errors:
     * Specify "manual" and explain the code change needed

3. **Sequential dependencies** (use "enabled" field):
   - If problem A enables problem B (B is in A's "enabled" list):
     * Create ONE strategy covering both
     * Explain in causal_chain why B appears after fixing A
   - Example: Config fix → validation now runs → reveals formatting errors

4. **Signals are OBSERVABLE PATTERNS that detect this failure**:

   WHAT SIGNALS ARE:
   - Actual error messages as they appear in CI logs
   - File patterns that indicate the problem
   - Tool names and versions
   - Error codes
   - Exit codes and command failures

   WHAT SIGNALS ARE NOT:
   ❌ "Error from L1: ..." (meta-reference)
   ❌ "File pattern: ..." (generic label)
   ❌ "Failed command: ..." (generic label)

   HOW TO EXTRACT SIGNALS:

   1. ERROR MESSAGE - Extract actual error as it appears:
      ✅ "mypy error [arg-type]: Argument 1 to joinpath has incompatible type Optional[Any]"
      ❌ "Error from L1: type mismatch"

   2. FILE INDICATOR - Actual file paths or patterns:
      ✅ "libs/agno/agno/workspace/helpers.py at line 42"
      ✅ "67 files matching docs/source/**/*.rst"
      ❌ "File pattern: helpers.py"

   3. COMMAND FAILURE - Exact command that fails:
      ✅ "Command 'python -m mypy .' exits with code 1"
      ✅ "pytest tests/ → 3 tests FAILED"
      ❌ "Failed command: mypy ."

   4. TOOL/VERSION - If relevant to detection:
      ✅ "mypy 1.8.0 reports incompatible types"
      ✅ "ruff 0.1.9 F401 unused import"
      ❌ "Validator: mypy"

   EXAMPLES of good signals:
   - "mypy: error: Incompatible return value type (got RunResponse, expected str) [return-value] at chain.py:84"
   - "pytest: AssertionError: assert 200 == 401 in tests/test_auth.py::test_login"
   - "ruff check: F401 'numpy.typing.DTypeLike' imported but unused in 5 *.py files"
   - "docstrfmt --check docs/source/ → 67 files with heading adornment errors"

5. **Key actions must be EXECUTABLE and INCLUDE AUTOMATION COMMANDS**:
   - Start with prerequisite checks
   - **For automation tools**: Include EXACT install and run commands from available tools
   - **For manual fixes**: Specify exact code changes
   - Include verification steps

   AUTOMATION TOOL GUIDANCE:
   - When using an automation tool, key_actions MUST include:
     1. Install command: "pip install <tool>" (from automation tools list)
     2. Run command: "python -m <tool> <args> <target>" (use actual file paths/patterns)
     3. Verification: Re-run the validator to confirm fix

   Examples:
   - AUTOMATION:
     * "Step 1: Install docformatter: pip install docformatter"
     * "Step 2: Run docformatter: python -m docformatter --in-place --recursive docs/source/"
     * "Step 3: Verify: python -m pytest tests/"

   - MANUAL:
     * "Step 1: Open libs/agno/agno/agent.py"
     * "Step 2: Add null check: if self.knowledge is not None:"
     * "Step 3: Verify: mypy libs/agno/agno/agent.py"

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
        //
        // PATTERN A: Config + Code fix (e.g., dependency issues)
        //   "Step 1: CONFIG - Open <config_file> and modify <section>: set <key>=<value>"
        //   "Step 2: CODE - Open <source_file> and change <specific_code_location>: <exact_change>"
        //   "Step 3: Verify: <verification_cmd from L1> (should pass)"
        //
        // PATTERN B: Automated fix (formatting/linting, many files)
        //   "Step 1: Install <tool>: <exact install_command from AUTOMATED_TOOLS>"
        //   "Step 2: Run automated fix: <exact fix_command with file paths> (fixes automatically, no manual check unless this fails)"
        //   "Step 3: Verify: <verification_cmd> (should pass)"
        //
        // PATTERN C: Manual fix only (semantic changes, few files)
        //   "Step 1: Open <actual file from L1> at line <line_number if available>"
        //   "Step 2: Change <old_code> to <new_code> because <reason from L1 fix_strategy>"
        //   "Step 3: Verify: <verification_cmd from L1> (should pass)"
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
