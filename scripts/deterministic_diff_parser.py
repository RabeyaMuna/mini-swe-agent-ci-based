#!/usr/bin/env python3
"""Compatibility wrapper for the shared deterministic diff parser."""

from utilities.deterministic_diff_parser import (
    chunk_structured_diff,
    format_structured_for_llm,
    parse_diff_to_structured,
    test_parser,
)

__all__ = [
    "chunk_structured_diff",
    "format_structured_for_llm",
    "parse_diff_to_structured",
    "test_parser",
]

if __name__ == "__main__":
    test_parser()
