"""
Ablation retrieval classes that disable specific components.

Each class inherits from STAIRRetrieval and overrides specific stages
to disable that component. This allows testing the contribution of
each component to overall repair performance.

All classes REUSE existing infrastructure - no core code changes.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from project
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_plugin.stair_retrieval import STAIRRetrieval


class NoDependencyExpansionRetrieval(STAIRRetrieval):
    """
    Disable dependency expansion (STAGE 4).

    This ablation tests whether identifying prerequisite problems
    (e.g., "fix linting before tests") improves repair success.

    REUSES: All other stages from parent STAIRRetrieval
    MODIFIES: Returns empty list from _stage_4_extract_related_problems
    """

    def _stage_4_extract_related_problems(self, match_sets, enriched_problems):
        """Override: Return empty list instead of extracting dependencies."""
        print("[Ablation] Dependency expansion DISABLED")
        return []  # No dependency problems


class NoRecurringRecallRetrieval(STAIRRetrieval):
    """
    Disable recurring-problem recall (STAGE 5).

    This ablation tests whether detecting common patterns across
    historical repairs improves success.

    REUSES: All other stages from parent STAIRRetrieval
    MODIFIES: Returns empty list from _stage_5_detect_common_patterns
    """

    def _stage_5_detect_common_patterns(self, query):
        """Override: Return empty list instead of detecting common patterns."""
        print("[Ablation] Recurring-problem recall DISABLED")
        return []  # No common problems


class NoApplicabilityFilterRetrieval(STAIRRetrieval):
    """
    Disable applicability filtering (STAGE 2).

    This ablation tests whether filtering retrieved memories by
    file/workflow relevance improves precision.

    REUSES: All other stages from parent STAIRRetrieval
    MODIFIES: Returns all matches without filtering
    """

    def _stage_2_per_problem_filtering(self, ci_problem, match_set, query):
        """Override: Return all matches without filtering."""
        print("[Ablation] Applicability filtering DISABLED")

        # Return all matches unfiltered
        return {
            "l1_matches": match_set.get("l1_matches", []),
            "l2_matches": match_set.get("l2_matches", []),
            "l3_matches": match_set.get("l3_matches", []),
        }


class NoOrderingRetrieval(STAIRRetrieval):
    """
    Disable dependency-based ordering (STAGE 6).

    This ablation tests whether ordering problems by dependencies
    and priority improves repair success.

    REUSES: All other stages from parent STAIRRetrieval
    MODIFIES: Returns problems in original order without reordering
    """

    def _stage_6_final_reorder(self, problems):
        """Override: Return problems in original order."""
        print("[Ablation] Dependency-based ordering DISABLED")
        return problems  # Keep original order


class NoHiddenProblemsRetrieval(STAIRRetrieval):
    """
    Disable BOTH dependency expansion AND recurring-problem recall.

    This tests the combined contribution of "hidden problem prediction"
    mechanisms that identify problems not directly visible in CI logs.

    REUSES: All other stages from parent STAIRRetrieval
    MODIFIES: Disables both STAGE 4 and STAGE 5
    """

    def _stage_4_extract_related_problems(self, match_sets, enriched_problems):
        """Override: No dependency expansion."""
        print("[Ablation] Dependency expansion DISABLED (hidden-problem ablation)")
        return []

    def _stage_5_detect_common_patterns(self, query):
        """Override: No recurring recall."""
        print("[Ablation] Recurring recall DISABLED (hidden-problem ablation)")
        return []


# Registry for easy lookup
ABLATION_RETRIEVALS = {
    "NoDependencyExpansionRetrieval": NoDependencyExpansionRetrieval,
    "NoRecurringRecallRetrieval": NoRecurringRecallRetrieval,
    "NoApplicabilityFilterRetrieval": NoApplicabilityFilterRetrieval,
    "NoOrderingRetrieval": NoOrderingRetrieval,
    "NoHiddenProblemsRetrieval": NoHiddenProblemsRetrieval,
}


def get_retrieval_class(name):
    """Get retrieval class by name."""
    if name in ABLATION_RETRIEVALS:
        return ABLATION_RETRIEVALS[name]
    else:
        raise ValueError(f"Unknown retrieval class: {name}. Available: {list(ABLATION_RETRIEVALS.keys())}")
