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

    prompt = f"""You are analyzing a cluster of SIMILAR problems to decide which should be MERGED.

VALIDATION: {validation_cmd}
CLUSTER SIZE: {cluster_size} problems

PROBLEMS:
{problems_text}

---

YOUR TASK: Group similar problems together.

MERGE CRITERIA (ALL must be true to merge):
OK Root causes express the SAME underlying issue
OK Fixes use the SAME pattern/approach
OK Files can be treated as "same problem in multiple places"
OK No interdependencies within cluster

KEEP SEPARATE if ANY true:
FAIL Root causes are DIFFERENT
FAIL Fixes require DIFFERENT approaches
FAIL Problems have different complexity levels
FAIL One problem is a configuration-key repair and another is a package,
     environment, resolver, API-adaptation, or deprecation/replacement repair with a different causal explanation

IMPORTANT: Be CONSERVATIVE. When in doubt, keep problems SEPARATE.

---

OUTPUT FORMAT (VALID JSON ONLY):

Return a JSON object with a "problems" array. Each entry represents ONE output problem:

{{
  "problems": [
    {{
      "merged_problems": [1, 2],
      "problem": "Unified description of what failed",
      "root_cause": "Unified root cause",
      "how_fixed": "Unified fix pattern",
      "why_fix_works": "Why this fix solves all merged instances"
    }},
    {{
      "merged_problems": [3],
      "problem": "Original problem 3 description",
      "root_cause": "Original problem 3 root cause",
      "how_fixed": "Original problem 3 fix",
      "why_fix_works": "Why this fix works for problem 3"
    }}
  ]
}}

RULES:
1. merged_problems with MULTIPLE IDs [1, 2] = These problems are MERGED into one
2. merged_problems with SINGLE ID [3] = This problem stays DISTINCT
3. EVERY input problem (1 through {cluster_size}) MUST appear in exactly ONE merged_problems array
4. Total problem IDs across all merged_problems arrays = {{1, 2, 3, ..., {cluster_size}}}

WRITING GUIDELINES:
- For "problem": Describe WHAT failed (not "Problem 1 is..." or "Problems 1 and 2...")
- For "root_cause": Describe WHY it failed (technical reason, not problem references)
- For "how_fixed": Describe the fix approach (not "Fixed problem 1 by...")
- For "why_fix_works": Explain why the fix solves the issue (technical explanation)
- Treat BEFORE as the pre-fix state and AFTER as the applied repair. Preserve
  that direction exactly; never reverse an addition/removal/replacement.
- Preserve every exact configuration key, package name, version constraint,
  source, extra/group, and old -> new value present in the input problems.
- If problems are merged, the merged problem/root_cause/how_fixed must retain
  the union of ALL input operations. A concise rewrite must not drop package or
  configuration details.
- Sharing the same manifest path or validation command is not a reason to merge
  independent configuration and package problems.

EXAMPLE 1 - All merged:
Input: Problems 1, 2, 3 all have same root cause
Output:
{{
  "problems": [
    {{
      "merged_problems": [1, 2, 3],
      "problem": "Import statements fail due to module not found errors",
      "root_cause": "Refactored code moved modules to new package structure without updating imports",
      "how_fixed": "Updated all import paths to reflect new package organization",
      "why_fix_works": "Corrects module resolution by pointing to actual file locations"
    }}
  ]
}}

EXAMPLE 2 - Partial merge:
Input: Problems 1+2 similar (type hints), Problem 3 different (import error)
Output:
{{
  "problems": [
    {{
      "merged_problems": [1, 2],
      "problem": "Type checking fails due to missing type annotations",
      "root_cause": "Function parameters lack type hints required by mypy",
      "how_fixed": "Added type annotations to all function parameters",
      "why_fix_works": "Satisfies mypy's static type checking requirements"
    }},
    {{
      "merged_problems": [3],
      "problem": "Import error for renamed module",
      "root_cause": "Module was renamed during refactoring but import statement not updated",
      "how_fixed": "Updated import statement to use new module name",
      "why_fix_works": "Resolves module not found error by using correct name"
    }}
  ]
}}

EXAMPLE 3 - All separate:
Input: Problems 1, 2, 3 all have different root causes
Output:
{{
  "problems": [
    {{
      "merged_problems": [1],
      "problem": "Missing type hint on return value",
      "root_cause": "Function missing return type annotation",
      "how_fixed": "Added return type annotation",
      "why_fix_works": "Satisfies mypy return type requirement"
    }},
    {{
      "merged_problems": [2],
      "problem": "Incorrect import path",
      "root_cause": "Import uses old package structure",
      "how_fixed": "Updated import to new package path",
      "why_fix_works": "Points to correct module location"
    }},
    {{
      "merged_problems": [3],
      "problem": "Missing dependency in requirements",
      "root_cause": "New code uses library not in requirements.txt",
      "how_fixed": "Added missing library to requirements",
      "why_fix_works": "Ensures library is installed"
    }}
  ]
}}

{strict_json_rules}

CRITICAL: You MUST complete the ENTIRE JSON response:
- Include ALL {cluster_size} problems in your output
- Close all arrays with ]
- Close all objects with }}
- Ensure every opening bracket has a closing bracket
- The last character should be }} for the outer object
- DO NOT stop mid-response!
"""

    return prompt
