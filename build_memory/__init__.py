"""
Memory building module for CI failure repair.

This module contains all logic for building three-level memory abstractions:
- L1: Problem-level concrete failures (file-specific, detailed)
- L2: Repair strategies (reusable patterns for fixing CI issues)
- L3: Universal cross-repo fix patterns (extracted from L1 + L2)
"""

from build_memory.build_l1 import (
    build_l1_memory,
    generate_l1_from_decomposed_problems,
    normalize_l1_record,
    create_search_document_l1,
)
from build_memory.build_l2 import (
    build_l2_memory,
    AUTOMATED_TOOLS,
)
from build_memory.build_l3 import (
    build_l3_memory,
    normalize_l3_record,
    merge_l3_records,
    create_search_document_l3,
)

__all__ = [
    # L1 builders
    "build_l1_memory",
    "generate_l1_from_decomposed_problems",
    "normalize_l1_record",
    "create_search_document_l1",
    # L2 builders
    "build_l2_memory",
    "AUTOMATED_TOOLS",
    # L3 builders
    "build_l3_memory",
    "normalize_l3_record",
    "merge_l3_records",
    "create_search_document_l3",
]
