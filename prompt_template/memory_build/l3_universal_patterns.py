"""
L3 Universal Patterns Prompts.

Prompts for generating universal fix patterns from L1 + L2.
"""

from typing import Dict
import json


def build_l3_prompt(l1_memory: Dict, l2_memory: Dict) -> str:
    """
    Build LLM prompt to generate L3 universal patterns from L1 + L2.

    Args:
        l1_memory: L1 memory with sequential problems
        l2_memory: L2 memory with repair strategies

    Returns:
        Prompt string for LLM
    """

    issue_id = l1_memory.get("issue_id")
    repo = l1_memory.get("repo")
    workflow = l1_memory.get("workflow")
    changed_files = l1_memory.get("changed_files", [])
    problems = l1_memory.get("problems", [])

    failure_identify = l2_memory.get("failure_identify", [])
    repair_strategies = l2_memory.get("repair_strategies", [])

    prompt = f"""You are extracting UNIVERSAL FIX PATTERNS (L3) from L1 + L2 data.

ISSUE: {issue_id}
REPO: {repo}
WORKFLOW: {workflow}

=== CHANGED FILES ===
{json.dumps(changed_files, indent=2)}

=== L1 PROBLEMS ({len(problems)} total) ===
{json.dumps(problems, indent=2)}

Each L1 problem has:
- problem_id, verification_cmd, failure_type
- problem: Error description
- root_cause: Why it occurred
- fix_strategy: How it was fixed
- files: Affected files
- enabled: Problems revealed after fixing this (dependencies)

=== L2 REPAIR STRATEGIES ===
Failure Summary: {json.dumps(failure_identify, indent=2)}

Strategies: {json.dumps(repair_strategies, indent=2)}

Each L2 strategy has:
- step, applies_to_failures, causal_chain
- summary, intent, reasoning, rationale
- when_to_apply, signals, key_actions, pitfalls

=== TASK ===
Extract 1-3 UNIVERSAL PATTERNS that can be reused across repos.

A universal pattern:
1. **Generalizes** the root cause (not repo-specific)
2. **Provides universal fix steps** (work for similar issues anywhere)
3. **Shows concrete examples** from L1 (before/after code)
4. **Identifies dependent changes** from L1's "enabled" relationships
5. **Extracts signals** from L2 that indicate when to apply
6. **Preserves automation guidance** from L2, including install/setup commands and exact run targets when a tool can fix the problem
7. **Adds Python cleanup guidance**: if a pattern changes Python files, universal_fix.steps must include Ruff autofix and formatting before final validation

EXAMPLES OF GOOD PATTERNS:

Pattern ID: "config_removed_plugin_breaks_types"
- Generalizable: Any plugin removal that breaks dependent types
- Universal fix: Replace plugin types with standard equivalents
- Applies to: numpy, pandas, django, any plugin-based typing

Pattern ID: "bulk_formatting_errors_after_linter_config_change"
- Generalizable: Config change → many uniform formatting errors
- Universal fix: Use automation tool for uniform mechanical fixes
- Applies to: docstrfmt, black, isort, any formatter

Pattern ID: "sequential_validator_dependency_chain"
- Generalizable: Early validator blocks later ones
- Universal fix: Fix validators in pipeline order
- Applies to: mypy → pytest → docstrfmt, any sequential CI pipeline

=== OUTPUT SCHEMA ===

For EACH universal pattern found, generate:

{{
  "pattern_id": "kebab-case-descriptive-name (e.g., 'numpy-plugin-type-removal')",

  "failure_type": "primary error type from L1 (e.g., 'type_checking', 'formatting')",

  "failure_pattern": "One-sentence universal description of what fails (project-agnostic)",

  "problem": "2-3 sentence explanation of WHY this pattern occurs universally.
            - Explain the root cause mechanism (not specific to this repo)
            - Use L1 root_cause but generalize it
            - Example: 'When a mypy plugin is removed from config, types provided by that plugin become undefined. Code using those types fails type checking because mypy no longer knows about them.'",

  "reasoning": "Evidence from L2 that shows this pattern:
               - Reference L2 strategies that demonstrate this
               - Show the pattern: mechanical vs semantic, root cause first, etc.
               - Example: 'EXTRACTED FROM L2: Strategy 1 shows config change (root cause) → type errors (effect). Strategy 2 shows 67 identical errors → automation. Pattern: Config changes are root causes; uniform errors enable automation.'",

  "when_to_apply": "Universal conditions when this pattern applies (not repo-specific):
                   - Describe the scenario in general terms
                   - Example: 'When a typing plugin or extension is removed from linter/type-checker configuration AND code uses types provided by that plugin.'",

  "signals": [
    "Observable indicators from L2, made universal:",
    "- Config file changed (pyproject.toml, setup.cfg, .ini)",
    "- Error messages reference undefined types or removed features",
    "- Failed validator: mypy, pylint, or other type checker",
    "- Multiple files fail with similar type errors",
    "DO NOT include repo-specific paths like 'py/flwr/common/typing.py'",
    "DO include patterns like: '*.py files in multiple directories', 'config removed [tool.X] section'"
  ],

  "universal_fix": {{
    "approach": "One-sentence universal approach (e.g., 'Replace plugin-specific types with standard Python equivalents')",

    "steps": [
      "Numbered universal steps (generalized from L2 key_actions):",
      "1. Identify what the removed config provided (check config diff)",
      "2. Find code using those features (grep for imports/types)",
      "3. Replace with standard equivalents (library docs for alternatives)",
      "4. Remove imports of plugin-specific features",
      "5. If Python files changed, run Ruff cleanup on affected files or their parent directory: pip install ruff; ruff check --fix <target>; ruff format <target>",
      "6. Verify with the validator (run type checker)",
      "",
      "DO NOT include repo-specific commands like 'python -m mypy py/'",
      "DO include generic patterns like 'Run type checker on affected modules'",
      "DO include automation commands when the repair is mechanical, such as installing the formatter/linter and running it against affected files or directories"
    ],

    "applies_to": [
      "Broader contexts where this fix works:",
      "- numpy.typing plugin types → standard types",
      "- pandas plugin types → standard types",
      "- django-stubs types → standard types",
      "- any library-specific typing plugin removal",
      "",
      "Make this BROAD - show the pattern applies beyond just this specific case"
    ]
  }},

  "examples": [
    {{
      "file": "path/from/L1 (keep actual path as evidence)",
      "before": "Code snippet BEFORE fix (extract from L1 if available, otherwise infer):
                from numpy.typing import DTypeLike
                def add(..., dtype: DTypeLike = np.int64):",
      "after": "Code snippet AFTER fix (extract from L1 if available, otherwise infer):
               from typing import Any
               def add(..., dtype: Any = np.int64):"
    }},
    {{
      "file": "another file from L1 showing same pattern",
      "before": "...",
      "after": "..."
    }}
  ],

  "dependent_changes": [
    "Problems that arise BECAUSE of this fix (from L1 'enabled' field):",
    "- After fixing config: test failures may be revealed (validator now runs)",
    "- After fixing types: formatting errors may appear (next pipeline stage)",
    "- Use L1 enabled relationships to identify cascades",
    "",
    "Format as: 'Problem type that emerges: why it emerges after this fix'",
    "Example: 'Formatting errors (docstrfmt): After type checking passes, formatting validator runs and reveals 67 RST heading errors'"
  ]
}}

=== EXTRACTION GUIDELINES ===

**Pattern Identification:**
1. Look for ROOT CAUSE patterns in L1 (config changes, breaking changes)
2. Look for MECHANICAL vs SEMANTIC patterns in L2 (automation decisions)
3. Look for DEPENDENCY patterns in L1 "enabled" fields (cascades)
4. Look for PIPELINE patterns in L2 causal_chain (sequential validation)

**Generalization:**
5. Remove repo-specific names (flower, flwr, framework)
6. Remove specific paths (py/flwr/common/typing.py → *.py files)
7. Remove specific commands (python -m mypy py/ → Run type checker)
8. Keep file patterns (*.rst, *.py, pyproject.toml)
9. Keep error patterns (undefined type, heading adornment)
10. Keep tool names (mypy, docstrfmt, black)

**Evidence:**
11. Pull examples from L1 problems (actual files that failed)
12. Extract signals from L2 repair_strategies
13. Use L2 reasoning to explain the pattern
14. Use L1 enabled to identify dependent_changes

**Quality:**
15. Pattern must apply to MULTIPLE repos (not just this one)
16. Universal_fix steps must work WITHOUT knowing this repo's structure
17. Signals must be observable in ANY repo with this pattern
18. Examples are EVIDENCE (keep specific) but fix is UNIVERSAL (generalize)
19. Automation-capable fixes must keep runnable commands in universal_fix.steps, including install/setup and the command target
20. Any universal_fix that changes *.py files must include Ruff cleanup before validation: install Ruff if needed, run `ruff check --fix <target>`, then run `ruff format <target>`

Return ONLY valid JSON:
{{
  "universal_patterns": [
    {{
      "pattern_id": "...",
      "failure_type": "...",
      "failure_pattern": "...",
      "problem": "...",
      "reasoning": "...",
      "when_to_apply": "...",
      "signals": [...],
      "universal_fix": {{}},
      "examples": [...],
      "dependent_changes": [...]
    }},
    ...
  ]
}}

CRITICAL:
- Generate 1-3 patterns (don't force more if data doesn't support it)
- Each pattern must be TRULY universal (reusable across repos)
- Examples stay specific (evidence), but fix steps are universal
- Dependent_changes come from L1 "enabled" relationships
"""

    return prompt
