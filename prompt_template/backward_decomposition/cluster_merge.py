"""
Cluster merge decision prompt.

Analyzes a cluster of similar problems to decide if they should be merged.
"""


def build_cluster_merge_prompt(
    validation_cmd: str,
    cluster_size: int,
    problems_text: str,
    strict_json_rules: str = "",
) -> str:
    """Build cluster merge decision prompt."""

    prompt = f"""You are analyzing a cluster of SIMILAR problems to decide if they should be MERGED.

VALIDATION: {validation_cmd}
CLUSTER SIZE: {cluster_size} problems

PROBLEMS:
{problems_text}

---

YOUR TASK: Decide ONE of these actions:

1. **MERGE** - All problems are essentially the SAME
   - Use when: Problems describe the same underlying issue in different files
   - Root causes are IDENTICAL (not just similar)
   - Fixes follow the EXACT SAME pattern
   - Example: "All files missing type hints for function parameters"

2. **PARTIAL_MERGE** - Some problems are same, others are distinct
   - Use when: Cluster has sub-groups of identical problems
   - Example: Problems 1,2,3 are same (merge), Problems 4,5 are different (separate)

3. **SEPARATE** - All problems are DISTINCT
   - Use when: Problems have different root causes or require different fixes
   - Even if similar, they're fundamentally different issues
   - Example: "Missing type hint" vs "Incorrect type hint"

MERGE CRITERIA (ALL must be true to merge):
✓ Root causes express the SAME underlying issue
✓ Fixes use the SAME pattern/approach
✓ Files can be treated as "same problem in multiple places"
✓ No interdependencies within cluster

SEPARATE if ANY true:
✗ Root causes are DIFFERENT
✗ Fixes require DIFFERENT approaches
✗ Problems have different complexity levels

OUTPUT VALID JSON ONLY:

For MERGE:
{{
  "action": "merge",
  "reasoning": "Why all problems are the same",
  "merged_problem": {{
    "problem": "Unified description of what failed",
    "root_cause": "Unified description of root cause",
    "how_fixed": "Unified description of fix pattern",
    "why_fix_works": "Why this fix solves all instances"
  }}
}}

For PARTIAL_MERGE:
{{
  "action": "partial_merge",
  "reasoning": "Why some merge, others separate",
  "sub_groups": [
    {{
      "problem_indices": [1, 2, 3],
      "action": "merge",
      "merged_problem": {{
        "problem": "...",
        "root_cause": "...",
        "how_fixed": "...",
        "why_fix_works": "..."
      }}
    }},
    {{
      "problem_indices": [4],
      "action": "separate",
      "reason": "Different root cause"
    }}
  ]
}}

For SEPARATE:
{{
  "action": "separate",
  "reasoning": "Why problems should stay separate"
}}

IMPORTANT: Be CONSERVATIVE. When in doubt, SEPARATE.

{strict_json_rules}
"""

    return prompt
