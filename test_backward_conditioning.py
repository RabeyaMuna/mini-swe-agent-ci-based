#!/usr/bin/env python3
"""
Test backward conditioning vs forward reasoning for L2 memory building.

This script demonstrates the difference between:
1. Forward reasoning: Analyze issue → guess what might have been done
2. Backward conditioning: Issue + verified patch → reason backward from ground truth

Usage:
    python test_backward_conditioning.py
"""

import json
from pathlib import Path

# Sample decomposed issue
SAMPLE_ISSUE = {
    "original_issue_id": "1",
    "sha_fail": "2c06ffa4c9d2c37846c60ad75899b4d72f214ff9",
    "repo": "huggingface/diffusers",
    "benchmark_ci_context": {
        "workflow_path": ".github/workflows/pr_quality.yml"
    },
    "atomic_problems": [
        {
            "order": 1,
            "issue_type": "Import Order - isort formatting",
            "ci_validation": "python -m isort --check-only .",
            "problem": "isort validation fails due to incorrect import ordering",
            "why": "Imports not sorted alphabetically according to isort configuration",
            "affected_files": [
                {
                    "file": "examples/community/ip_adapter_face_id.py",
                    "role": "primary",
                    "what_is_wrong": "safetensors imported before torch (should be alphabetical)",
                    "how_to_fix": "Reorder imports alphabetically"
                }
            ]
        }
    ]
}

# Verified patch from generated_patches_success_only.json
VERIFIED_PATCH = """diff --git a/examples/community/ip_adapter_face_id.py b/examples/community/ip_adapter_face_id.py
index e3c5a2c84..d9325742c 100644
--- a/examples/community/ip_adapter_face_id.py
+++ b/examples/community/ip_adapter_face_id.py
@@ -14,12 +14,12 @@

 import inspect
 from typing import Any, Callable, Dict, List, Optional, Union
-from safetensors import safe_open

 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 from packaging import version
+from safetensors import safe_open
 from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection"""


def show_forward_vs_backward():
    """Show the difference in prompts between forward and backward conditioning."""

    print("="*80)
    print("BACKWARD CONDITIONING TEST")
    print("="*80)
    print()

    # FORWARD REASONING PROMPT
    print("❌ FORWARD REASONING (Current - speculative)")
    print("-" * 80)
    forward_prompt = f"""Analyze this CI failure and create L2 memory.

ISSUE: Import order validation failed
PROBLEM: isort validation fails due to incorrect import ordering

Create repair trajectory showing what was likely done to fix this.
"""
    print(forward_prompt)
    print()
    print("RISK: LLM might invent speculative steps like:")
    print("  - 'Developer manually reordered imports'")
    print("  - 'Used automated tool to fix'")
    print("  - 'Checked Python style guide'")
    print("  → These may not reflect what ACTUALLY happened!")
    print()

    # BACKWARD CONDITIONING PROMPT
    print("✅ BACKWARD CONDITIONING (Enhanced - ground truth)")
    print("-" * 80)
    backward_prompt = f"""Analyze this CI failure and create L2 memory.

ISSUE: Import order validation failed
PROBLEM: isort validation fails due to incorrect import ordering

VERIFIED PATCH (GROUND TRUTH - what ACTUALLY fixed the issue):
```diff
{VERIFIED_PATCH}
```

CRITICAL: Work BACKWARD from this verified patch.
- All reasoning must be CONSISTENT with what the patch actually changed
- The patch shows: safetensors import moved AFTER torch imports
- This is NOT a general "reorder imports" - it's SPECIFIC: move safetensors down
- Do not invent speculative steps that don't align with the verified fix

For file_changes, explain:
1. WHY this file? → Because patch modifies examples/community/ip_adapter_face_id.py
2. WHAT changed? → safetensors import moved from line 17 to line 22 (after version)
3. WHY this works? → Places safetensors in correct alphabetical position
"""
    print(backward_prompt)
    print()
    print("BENEFITS:")
    print("  ✓ LLM reasoning is CONSTRAINED by actual patch")
    print("  ✓ No speculation about what 'might' have been done")
    print("  ✓ Captures ACTUAL fix pattern (move safetensors, not 'reorder all')")
    print("  ✓ More accurate, transferable knowledge")
    print()

    # SHOW EXPECTED OUTPUT DIFFERENCE
    print("="*80)
    print("EXPECTED OUTPUT DIFFERENCE")
    print("="*80)
    print()

    print("❌ FORWARD (might generate):")
    print("-" * 80)
    forward_output = {
        "atomic_problems": [{
            "problem_id": 1,
            "issue_type": "Import Order - isort formatting",
            "failed_cmd": "python -m isort --check-only .",
            "problem": "isort validation fails due to incorrect import ordering",
            "file_changes": [{
                "file": "examples/community/ip_adapter_face_id.py",
                "fix": "Reorder imports to be alphabetical"  # ← VAGUE
            }]
        }],
        "repair_trajectory_summary": "Fixed import ordering to comply with isort"  # ← VAGUE
    }
    print(json.dumps(forward_output, indent=2))
    print()
    print("PROBLEM: Too vague - doesn't capture WHAT specifically was reordered!")
    print()

    print("✅ BACKWARD (will generate):")
    print("-" * 80)
    backward_output = {
        "atomic_problems": [{
            "problem_id": 1,
            "issue_type": "Import Order - isort formatting",
            "failed_cmd": "python -m isort --check-only .",
            "problem": "isort validation fails because safetensors import appears before torch/packaging imports, violating alphabetical order",
            "file_changes": [{
                "file": "examples/community/ip_adapter_face_id.py",
                "fix": "Move 'from safetensors import safe_open' from line 17 to line 22, placing it after 'from packaging import version' to maintain alphabetical order"  # ← SPECIFIC
            }]
        }],
        "repair_trajectory_summary": "Moved safetensors import to correct alphabetical position (after packaging, before transformers) to satisfy isort validation in ip_adapter_face_id.py"  # ← SPECIFIC
    }
    print(json.dumps(backward_output, indent=2))
    print()
    print("BENEFIT: Captures EXACT pattern - move safetensors after packaging!")
    print("  → Agent learning this can apply same pattern to similar issues")
    print()

    # TRANSFERABILITY EXAMPLE
    print("="*80)
    print("TRANSFERABILITY EXAMPLE")
    print("="*80)
    print()

    print("New issue in different repo:")
    print("  Error: 'from requests import get' appears before 'import os'")
    print()

    print("❌ With FORWARD memory:")
    print("  Agent retrieves: 'Reorder imports to be alphabetical'")
    print("  → Too vague, agent doesn't know HOW to reorder")
    print()

    print("✅ With BACKWARD memory:")
    print("  Agent retrieves: 'Move safetensors after packaging to maintain alphabetical order'")
    print("  → Pattern: third-party imports after stdlib, alphabetical within groups")
    print("  → Agent can apply: Move 'requests' import after 'os' import")
    print("  → ACTIONABLE!")
    print()


def show_implementation_simplicity():
    """Show how simple the implementation change is."""

    print("="*80)
    print("IMPLEMENTATION: MINIMAL CHANGES REQUIRED")
    print("="*80)
    print()

    print("STEP 1: Load verified patches (one-time)")
    print("-" * 80)
    print("""
# In main() function:
patches_path = Path("data/generated_patches_success_only.json")
verified_patches = _load_verified_patches(patches_path)
# Returns: {sha_fail: diff}
""")

    print()
    print("STEP 2: Pass patch to _build_l2_with_llm (one parameter)")
    print("-" * 80)
    print("""
# Before:
l2_result = _build_l2_with_llm(issue, llm)

# After:
verified_patch = verified_patches.get(issue["sha_fail"])
l2_result = _build_l2_with_llm(issue, llm, verified_patch=verified_patch)
""")

    print()
    print("STEP 3: Enhanced prompt (one if-statement)")
    print("-" * 80)
    print("""
# In _build_l2_with_llm():
backward_conditioning_section = ""
if verified_patch:
    backward_conditioning_section = f'''
VERIFIED PATCH (GROUND TRUTH):
```diff
{verified_patch}
```

Work BACKWARD from this patch:
- Why was THIS file selected?
- What EXACTLY changed?
- Why does THIS fix work?
'''

prompt = f'''...{backward_conditioning_section}...'''
""")

    print()
    print("THAT'S IT! 3 simple changes:")
    print("  1. Load patches (5 lines)")
    print("  2. Pass patch parameter (1 line)")
    print("  3. Add to prompt if patch exists (6 lines)")
    print()
    print("TOTAL: ~15 lines of code added!")
    print()


if __name__ == "__main__":
    show_forward_vs_backward()
    print()
    show_implementation_simplicity()

    print("="*80)
    print("USAGE")
    print("="*80)
    print()
    print("Run with backward conditioning (default):")
    print("  python scripts/build_memory_from_decomposed.py \\")
    print("    --decomposed data/trs/decomposed_issues.json \\")
    print("    --verified-patches data/generated_patches_success_only.json")
    print()
    print("Run without backward conditioning (forward only):")
    print("  python scripts/build_memory_from_decomposed.py \\")
    print("    --decomposed data/trs/decomposed_issues.json \\")
    print("    --no-backward-conditioning")
    print()
    print("="*80)
    print("EXPECTED IMPROVEMENT: +8-12% Pass@1 (from ConRAD paper)")
    print("="*80)
