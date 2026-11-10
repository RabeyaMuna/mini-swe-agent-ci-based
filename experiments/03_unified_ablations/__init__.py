"""
Unified Ablations Experiment

Component-level ablation study covering:
- Reviewer jcJU Q2: Component ablation (4 configs on 50 instances)
- Reviewer u2Pr Q1: Hidden-problem prediction (combined on 100 instances)

All ablations REUSE existing ReCAP infrastructure via inheritance.
"""

from .ablation_configs import ABLATION_CONFIGS, get_configs_to_run, get_configs_to_reuse
from .ablation_retrievals import (
    NoDependencyExpansionRetrieval,
    NoRecurringRecallRetrieval,
    NoApplicabilityFilterRetrieval,
    NoOrderingRetrieval,
    NoHiddenProblemsRetrieval,
    get_retrieval_class,
)

__all__ = [
    "ABLATION_CONFIGS",
    "get_configs_to_run",
    "get_configs_to_reuse",
    "NoDependencyExpansionRetrieval",
    "NoRecurringRecallRetrieval",
    "NoApplicabilityFilterRetrieval",
    "NoOrderingRetrieval",
    "NoHiddenProblemsRetrieval",
    "get_retrieval_class",
]
