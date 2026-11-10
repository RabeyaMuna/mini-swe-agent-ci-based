#!/usr/bin/env python3
"""
Quick test to verify ablation retrieval classes work correctly.

Tests:
1. All wrapper classes can be instantiated
2. They inherit from STAIRRetrieval
3. They override the correct methods
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_retrievals import (
    NoDependencyExpansionRetrieval,
    NoRecurringRecallRetrieval,
    NoApplicabilityFilterRetrieval,
    NoOrderingRetrieval,
    NoHiddenProblemsRetrieval,
    get_retrieval_class,
)
from memory_plugin.stair_retrieval import STAIRRetrieval


def test_inheritance():
    """Test all wrappers inherit from STAIRRetrieval."""
    classes = [
        NoDependencyExpansionRetrieval,
        NoRecurringRecallRetrieval,
        NoApplicabilityFilterRetrieval,
        NoOrderingRetrieval,
        NoHiddenProblemsRetrieval,
    ]

    for cls in classes:
        assert issubclass(cls, STAIRRetrieval), f"{cls.__name__} doesn't inherit from STAIRRetrieval"
        print(f"✓ {cls.__name__} inherits from STAIRRetrieval")


def test_method_overrides():
    """Test each wrapper overrides the correct methods."""
    # Test NoDependencyExpansionRetrieval
    assert hasattr(NoDependencyExpansionRetrieval, '_stage_4_extract_related_problems')
    print("✓ NoDependencyExpansionRetrieval overrides _stage_4_extract_related_problems")

    # Test NoRecurringRecallRetrieval
    assert hasattr(NoRecurringRecallRetrieval, '_stage_5_detect_common_patterns')
    print("✓ NoRecurringRecallRetrieval overrides _stage_5_detect_common_patterns")

    # Test NoApplicabilityFilterRetrieval
    assert hasattr(NoApplicabilityFilterRetrieval, '_stage_2_per_problem_filtering')
    print("✓ NoApplicabilityFilterRetrieval overrides _stage_2_per_problem_filtering")

    # Test NoOrderingRetrieval
    assert hasattr(NoOrderingRetrieval, '_stage_6_final_reorder')
    print("✓ NoOrderingRetrieval overrides _stage_6_final_reorder")

    # Test NoHiddenProblemsRetrieval
    assert hasattr(NoHiddenProblemsRetrieval, '_stage_4_extract_related_problems')
    assert hasattr(NoHiddenProblemsRetrieval, '_stage_5_detect_common_patterns')
    print("✓ NoHiddenProblemsRetrieval overrides both dependency and recurring stages")


def test_get_retrieval_class():
    """Test retrieval class registry."""
    cls = get_retrieval_class("NoDependencyExpansionRetrieval")
    assert cls == NoDependencyExpansionRetrieval
    print("✓ get_retrieval_class() works")

    try:
        get_retrieval_class("NonExistentClass")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ get_retrieval_class() raises ValueError for unknown classes")


def test_instantiation():
    """Test each class can be instantiated (baseline mode)."""
    classes = [
        NoDependencyExpansionRetrieval,
        NoRecurringRecallRetrieval,
        NoApplicabilityFilterRetrieval,
        NoOrderingRetrieval,
        NoHiddenProblemsRetrieval,
    ]

    for cls in classes:
        # Instantiate in baseline mode (no memory loading)
        instance = cls(
            memory_dir="data/bidirect_trs",
            baseline_mode=True,  # Avoid loading memory files
        )
        assert isinstance(instance, STAIRRetrieval)
        print(f"✓ {cls.__name__} can be instantiated")


def main():
    print("Testing ablation retrieval classes...\n")

    try:
        test_inheritance()
        print()
        test_method_overrides()
        print()
        test_get_retrieval_class()
        print()
        test_instantiation()
        print()
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
