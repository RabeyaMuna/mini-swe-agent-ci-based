#!/usr/bin/env python3
"""
Test SEQUENTIAL backward conditioning for multi-validation CI failures.

Shows how backward conditioning should work WITH validation order awareness.
"""

import json

# Real example: flower_121 with 2 sequential validations
FLOWER_121_ISSUE = {
    "original_issue_id": "121",
    "sha_fail": "flower_sha_121",
    "repo": "flower",
    "validation_sequence": [
        {
            "order": 1,
            "validation_name": "mypy type checking",
            "command": "python -m mypy py",
            "blocks": [2]
        },
        {
            "order": 2,
            "validation_name": "mdformat documentation formatting",
            "command": "python -m mdformat --check --number docs/source",
            "depends_on": [1]
        }
    ],
    "atomic_problems": [
        {
            "order": 1,
            "issue_type": "Type Checking - Private API import",
            "ci_validation": "python -m mypy py",
            "problem": "mypy fails on numpy._typing import",
            "affected_files": ["framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py"]
        },
        {
            "order": 2,
            "issue_type": "Documentation Formatting - RST overlines",
            "ci_validation": "python -m mdformat --check --number docs/source",
            "problem": "mdformat fails on 88 RST files",
            "affected_files": ["framework/docs/source/*.rst (88 files)"]
        }
    ]
}

# Verified patch (combined - shows BOTH fixes)
VERIFIED_PATCH_COMBINED = """diff --git a/framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py b/framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py
index abc123..def456 100644
--- a/framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py
+++ b/framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py
@@ -5,7 +5,6 @@
 import numpy as np
-from numpy._typing import DTypeLike

 def add_arrays(
-    dtype: DTypeLike,
+    dtype: np.dtype[Any] | type[Any],
 ) -> NDArray:
     pass

diff --git a/framework/docs/source/tutorial.rst b/framework/docs/source/tutorial.rst
index 111222..333444 100644
--- a/framework/docs/source/tutorial.rst
+++ b/framework/docs/source/tutorial.rst
@@ -1,3 +1,4 @@
+=====
 Tutorial
 =====

[... and 87 more RST files with same pattern ...]
"""


def show_wrong_backward_conditioning():
    """Show what WRONG backward conditioning looks like (ignores sequence)."""

    print("="*80)
    print("❌ WRONG: Backward Conditioning WITHOUT Sequential Awareness")
    print("="*80)
    print()

    wrong_output = {
        "atomic_problems": [
            {
                "problem_id": 1,
                "issue_type": "Type Checking + Documentation Formatting",
                "problem": "mypy fails and mdformat fails",
                "file_changes": [
                    {
                        "file": "framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py",
                        "fix": "Replace numpy._typing.DTypeLike with np.dtype[Any]"
                    },
                    {
                        "file": "framework/docs/source/*.rst (88 files)",
                        "fix": "Add RST overlines to all headings"
                    }
                ]
            }
        ],
        "repair_trajectory_summary": "Fixed type checking and documentation formatting issues"
    }

    print(json.dumps(wrong_output, indent=2))
    print()
    print("PROBLEMS:")
    print("  ❌ Lumps TWO validations into ONE problem")
    print("  ❌ Doesn't show validation 1 BLOCKED validation 2")
    print("  ❌ Doesn't show mdformat error was HIDDEN until mypy passed")
    print("  ❌ Agent can't learn sequential repair strategy")
    print()


def show_correct_sequential_backward_conditioning():
    """Show what CORRECT sequential backward conditioning looks like."""

    print("="*80)
    print("✅ CORRECT: Sequential Backward Conditioning")
    print("="*80)
    print()

    correct_output = {
        "atomic_problems": [
            {
                "problem_id": 1,
                "validation_order": 1,
                "depends_on_validations": [],
                "blocks_validations": [2],

                "issue_type": "Type Checking - Private API import",
                "failed_cmd": "python -m mypy py",
                "exit_code": 1,
                "problem": "mypy type checker fails because ndarrays_arithmetic.py imports numpy._typing.DTypeLike (private API without type stubs)",

                "file_changes": [
                    {
                        "file": "framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py",
                        "fix": "Replace numpy._typing.DTypeLike with np.dtype[Any] | type[Any]",
                        "why_this_file": "Verified patch shows this file changed - removed numpy._typing import, replaced DTypeLike annotations",
                        "validation_fixed": 1
                    }
                ],

                "repair_trajectory": {
                    "investigation_steps": [
                        {
                            "step": 1,
                            "action": "python -m mypy py",
                            "observation": "Exit code 1: Skipping analyzing 'numpy._typing'",
                            "reasoning": "Validation order=1 blocks all subsequent. Must fix THIS FIRST."
                        }
                    ],
                    "fix_decisions": [
                        {
                            "step": 2,
                            "considered_approaches": [
                                {
                                    "approach": "Import from numpy.typing",
                                    "rejected": True,
                                    "rejected_because": "Not in patch - patch uses np.dtype[Any]"
                                },
                                {
                                    "approach": "Use np.dtype[Any] | type[Any]",
                                    "selected": True,
                                    "selected_because": "VERIFIED PATCH shows this approach was used"
                                }
                            ]
                        }
                    ],
                    "verification_steps": [
                        {
                            "step": 3,
                            "action": "python -m mypy py",
                            "expected_exit_code": 0,
                            "actual_exit_code": 0,
                            "outcome": "success",
                            "unblocks": "validation_2_mdformat"
                        }
                    ]
                },

                "verification_result": {
                    "validation_cmd": "python -m mypy py",
                    "exit_code": 0,
                    "status": "PASS",
                    "unblocks_validations": [2],
                    "next_validation_revealed": "validation_2_mdformat"
                }
            },

            {
                "problem_id": 2,
                "validation_order": 2,
                "depends_on_validations": [1],
                "blocks_validations": [],
                "revealed_after": "validation_1_passed",

                "issue_type": "Documentation Formatting - RST overlines",
                "failed_cmd": "python -m mdformat --check --number docs/source",
                "exit_code": 1,
                "problem": "mdformat fails on 88 RST files because top-level headings lack overlines",

                "file_changes": [
                    {
                        "file_pattern": "framework/docs/source/*.rst",
                        "files_affected_count": 88,
                        "fix": "Add overline adornment (=) above each top-level heading",
                        "why_these_files": "Verified patch shows 88 RST files modified with same pattern (add overline)",
                        "validation_fixed": 2
                    }
                ],

                "repair_trajectory": {
                    "investigation_steps": [
                        {
                            "step": 1,
                            "action": "python -m mdformat --check --number docs/source",
                            "observation": "88 files fail: 'Heading level 1 should have overline'",
                            "reasoning": "This validation COULD NOT RUN until validation 1 passed. Error was HIDDEN."
                        }
                    ],
                    "fix_decisions": [
                        {
                            "step": 2,
                            "considered_approaches": [
                                {
                                    "approach": "Manual edit 88 files",
                                    "rejected": True,
                                    "rejected_because": "Error-prone, but patch doesn't show automation evidence"
                                },
                                {
                                    "approach": "Bulk sed script",
                                    "selected": True,
                                    "selected_because": "VERIFIED PATCH shows uniform change across 88 files - indicates scripted fix"
                                }
                            ]
                        }
                    ],
                    "verification_steps": [
                        {
                            "step": 3,
                            "action": "python -m mdformat --check --number docs/source",
                            "expected_exit_code": 0,
                            "actual_exit_code": 0,
                            "outcome": "success",
                            "all_validations_complete": True
                        }
                    ]
                },

                "verification_result": {
                    "validation_cmd": "python -m mdformat --check --number docs/source",
                    "exit_code": 0,
                    "status": "PASS",
                    "unblocks_validations": [],
                    "all_validations_complete": True
                }
            }
        ],

        "sequential_workflow_metadata": {
            "total_validations": 2,
            "validation_dependency_chain": {
                "1": {
                    "name": "mypy_type_checking",
                    "blocks": [2],
                    "depends_on": [],
                    "files_from_patch": ["framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py"]
                },
                "2": {
                    "name": "mdformat_documentation_check",
                    "blocks": [],
                    "depends_on": [1],
                    "files_from_patch": ["framework/docs/source/*.rst (88 files)"]
                }
            },
            "repair_order_sequence": [
                "Step 1: Validation 1 (mypy) failed → saw error in ndarrays_arithmetic.py",
                "Step 2: Fixed ndarrays_arithmetic.py (from VERIFIED PATCH: replaced numpy._typing)",
                "Step 3: Validation 1 passed → REVEALED validation 2 (mdformat)",
                "Step 4: Saw mdformat error (88 RST files missing overlines)",
                "Step 5: Fixed 88 RST files (from VERIFIED PATCH: added overlines)",
                "Step 6: Validation 2 passed → ALL PASS ✓"
            ]
        },

        "repair_trajectory_summary": "SEQUENTIAL REPAIR: First fixed mypy (ndarrays_arithmetic.py - numpy._typing → np.dtype[Any]) which unblocked validation 1. This revealed mdformat failures (88 RST files). Then fixed RST overlines, passing validation 2. Total: 2 validations fixed in dependency order."
    }

    print(json.dumps(correct_output, indent=2))
    print()
    print("BENEFITS:")
    print("  ✅ Shows TWO distinct validations (not lumped together)")
    print("  ✅ Shows validation 1 BLOCKS validation 2")
    print("  ✅ Shows mdformat error was HIDDEN until mypy passed")
    print("  ✅ Maps patch changes to WHICH validation they fix")
    print("  ✅ Shows SEQUENTIAL repair timeline")
    print("  ✅ Agent learns: 'Fix blocker first → reveals next → fix next'")
    print()


def show_how_llm_should_reason():
    """Show the reasoning process LLM should follow."""

    print("="*80)
    print("LLM REASONING PROCESS (Sequential Backward Conditioning)")
    print("="*80)
    print()

    print("STEP 1: Parse validation sequence")
    print("-" * 80)
    print("""
Validation sequence:
1. mypy (blocks: 2)
2. mdformat (depends_on: 1)

→ Validation 1 is the blocker, validation 2 runs AFTER validation 1 passes
""")

    print()
    print("STEP 2: Parse verified patch - identify file groups")
    print("-" * 80)
    print("""
Verified patch shows TWO file groups:
Group A: framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py
         Change: numpy._typing.DTypeLike → np.dtype[Any] | type[Any]

Group B: framework/docs/source/*.rst (88 files)
         Change: Add '=====' above each 'Title\n====='

→ Two distinct fix patterns = Two distinct validations fixed
""")

    print()
    print("STEP 3: Map file groups to validations")
    print("-" * 80)
    print("""
Question: Which validation does Group A fix?
- Validation 1 command: python -m mypy py
- Group A file: ndarrays_arithmetic.py (Python file)
- Group A change: Type annotation fix (DTypeLike → np.dtype)
→ Group A fixes VALIDATION 1 (mypy type checking)

Question: Which validation does Group B fix?
- Validation 2 command: python -m mdformat --check docs/source
- Group B files: docs/source/*.rst (documentation)
- Group B change: RST heading format
→ Group B fixes VALIDATION 2 (mdformat)
""")

    print()
    print("STEP 4: Reconstruct sequential timeline")
    print("-" * 80)
    print("""
Timeline (working BACKWARD from patch):
1. FINAL STATE (patch): Both Group A and Group B changes applied
2. BEFORE Group B: Only Group A applied, validation 1 passed, validation 2 failed
3. BEFORE Group A: Both validations failed, but validation 2 didn't run yet
4. INITIAL STATE: Validation 1 failed (blocker), validation 2 hidden

Sequential repair (FORWARD in time):
1. Developer saw validation 1 error (mypy)
2. Applied Group A fix (ndarrays_arithmetic.py)
3. Validation 1 passed → validation 2 NOW RUNS
4. Developer saw validation 2 error (mdformat) - this was HIDDEN before
5. Applied Group B fix (88 RST files)
6. Validation 2 passed → issue resolved
""")

    print()
    print("STEP 5: Generate sequential L2 structure")
    print("-" * 80)
    print("""
atomic_problems: [
  {
    problem_id: 1,
    validation_order: 1,
    file_changes: [Group A files ONLY],  ← Not Group B!
    verification_result: { unblocks_validations: [2] }
  },
  {
    problem_id: 2,
    validation_order: 2,
    depends_on_validations: [1],
    revealed_after: "validation_1_passed",
    file_changes: [Group B files ONLY]  ← Not Group A!
  }
]

repair_trajectory_summary: "Sequential repair: Fixed validation 1 (Group A) → revealed validation 2 → fixed validation 2 (Group B)"
""")


if __name__ == "__main__":
    show_wrong_backward_conditioning()
    print()
    show_correct_sequential_backward_conditioning()
    print()
    show_how_llm_should_reason()

    print()
    print("="*80)
    print("KEY INSIGHT")
    print("="*80)
    print()
    print("Verified patch shows FINAL state (all fixes combined).")
    print("But the repair happened SEQUENTIALLY (one validation at a time).")
    print()
    print("LLM must:")
    print("  1. Parse validation sequence (which blocks which)")
    print("  2. Parse patch (identify distinct file groups/change patterns)")
    print("  3. Map each file group to the validation it fixes")
    print("  4. Reconstruct sequential timeline")
    print()
    print("This gives agent the knowledge:")
    print("  'When validation X blocks validation Y, fix X first to reveal Y'")
    print()
