"""
Configuration for unified ablation experiments.

Defines which configurations to run and which to reuse from existing results.
"""

ABLATION_CONFIGS = {
    # ========== REUSE EXISTING RESULTS ==========
    "baseline": {
        "description": "No memory (decomposition only)",
        "reuse_existing": True,
        "existing_path": "results/miniswe-agent/baseline_deepseek_v4_flash",
        "run_new": False,
    },
    "full_recap": {
        "description": "Full ReCAP (L1+L2+L3, bidirectional)",
        "reuse_existing": True,
        "existing_path": "results/miniswe-agent/bidirectional/l1_l2_l3_deepseek_v4_flash",
        "run_new": False,
    },

    # ========== NEW CONFIGURATIONS TO RUN ==========
    "no_dependency_expansion": {
        "description": "Disable dependency expansion (STAGE 4)",
        "reuse_existing": False,
        "run_new": True,
        "retrieval_class": "NoDependencyExpansionRetrieval",
        "memory_dir": "data/bidirect_trs",  # Reuse existing memory
        "ablation": "L1+L2+L3",
    },
    "no_recurring_recall": {
        "description": "Disable recurring-problem recall (STAGE 5)",
        "reuse_existing": False,
        "run_new": True,
        "retrieval_class": "NoRecurringRecallRetrieval",
        "memory_dir": "data/bidirect_trs",
        "ablation": "L1+L2+L3",
    },
    "no_applicability_filter": {
        "description": "Disable applicability filtering (STAGE 2)",
        "reuse_existing": False,
        "run_new": True,
        "retrieval_class": "NoApplicabilityFilterRetrieval",
        "memory_dir": "data/bidirect_trs",
        "ablation": "L1+L2+L3",
    },
    "no_ordering": {
        "description": "Disable dependency-based ordering (STAGE 6)",
        "reuse_existing": False,
        "run_new": True,
        "retrieval_class": "NoOrderingRetrieval",
        "memory_dir": "data/bidirect_trs",
        "ablation": "L1+L2+L3",
    },
    "no_hidden_problems": {
        "description": "Disable both dependency expansion AND recurring recall",
        "reuse_existing": False,
        "run_new": True,
        "retrieval_class": "NoHiddenProblemsRetrieval",
        "memory_dir": "data/bidirect_trs",
        "ablation": "L1+L2+L3",
    },
}


def get_configs_to_run():
    """Get list of configurations that need new runs."""
    return {k: v for k, v in ABLATION_CONFIGS.items() if v["run_new"]}


def get_configs_to_reuse():
    """Get list of configurations that reuse existing results."""
    return {k: v for k, v in ABLATION_CONFIGS.items() if v["reuse_existing"]}
