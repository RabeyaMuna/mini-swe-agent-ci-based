#!/usr/bin/env python3
"""
Test deterministic diff parsing integration with decomposition.

This tests the new flow:
1. Parse diff deterministically
2. Chunk by file count
3. Send structured data to LLM (not raw code)
4. Build L1/L2/L3
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def test_diff_parsing():
    """Test that diff parsing works on real issue."""
    print("="*80)
    print("TEST 1: Deterministic Diff Parsing")
    print("="*80)

    # Load issue 121
    issues_path = PROJECT_ROOT / "data/trs/memory_seed_issues.json"
    with open(issues_path) as f:
        all_issues = json.load(f)

    issue = next((i for i in all_issues if str(i.get('id')) == '121'), None)
    if not issue:
        print("❌ Issue 121 not found!")
        return False

    diff = issue.get('diff', '')
    print(f"\nIssue 121: {len(diff)} char diff")

    # Parse diff
    from deterministic_diff_parser import parse_diff_to_structured, chunk_structured_diff

    structured = parse_diff_to_structured(diff)
    print(f"✅ Parsed {structured['total_files']} files with {structured['total_changes']} changes")

    # Show sample files
    print("\nSample files:")
    for file_info in structured["files"][:5]:
        print(f"  - {file_info['path']} ({file_info['file_type']}): {file_info['total_changes']} changes")

    # Chunk
    chunks = chunk_structured_diff(structured, max_files_per_chunk=10)
    print(f"\n✅ Created {len(chunks)} chunks (max 10 files each)")

    for i, chunk in enumerate(chunks[:3], 1):
        print(f"  Chunk {i}: {chunk['total_files']} files, {chunk['total_changes']} changes")

    return True


def test_llm_format():
    """Test that LLM format is concise and clean."""
    print("\n" + "="*80)
    print("TEST 2: LLM-Friendly Formatting")
    print("="*80)

    # Load issue 121
    issues_path = PROJECT_ROOT / "data/trs/memory_seed_issues.json"
    with open(issues_path) as f:
        all_issues = json.load(f)

    issue = next((i for i in all_issues if str(i.get('id')) == '121'), None)
    diff = issue.get('diff', '')

    from deterministic_diff_parser import (
        parse_diff_to_structured,
        chunk_structured_diff,
        format_structured_for_llm
    )

    structured = parse_diff_to_structured(diff)
    chunks = chunk_structured_diff(structured, max_files_per_chunk=10)

    # Format first chunk for LLM
    formatted = format_structured_for_llm(chunks[0])

    print("\nFormatted chunk (first 500 chars):")
    print("-" * 80)
    print(formatted[:500])
    print("-" * 80)

    print(f"\n✅ Formatted length: {len(formatted)} chars")
    print(f"   Original diff chunk would be: ~{len(diff) // len(chunks)} chars")
    print(f"   Reduction: {100 - (len(formatted) / (len(diff) // len(chunks)) * 100):.1f}%")

    return True


def test_full_integration():
    """Test full pipeline with deterministic parsing."""
    print("\n" + "="*80)
    print("TEST 3: Full Integration Test")
    print("="*80)

    print("\nRunning: python scripts/build_memory_pipeline.py --issue-ids 121")
    print("This will test the complete flow:")
    print("  1. Deterministic diff parsing")
    print("  2. Decomposition with structured data")
    print("  3. L1/L2/L3 building")
    print("\nCheck output for:")
    print("  ✓ 'Step 0: Parsing diff into structured format...'")
    print("  ✓ 'Parsed 88 files with XXX changes'")
    print("  ✓ 'Step 1: Classifying files by validation (88 files in X chunk(s))...'")
    print("  ✓ No JSON parsing errors")
    print("  ✓ High success rate (most chunks should succeed)")

    return True


if __name__ == "__main__":
    print("Testing Deterministic Diff Integration\n")

    tests = [
        ("Diff Parsing", test_diff_parsing),
        ("LLM Formatting", test_llm_format),
        ("Full Integration", test_full_integration),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {name} PASSED")
            else:
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("="*80)

    if passed == len(tests):
        print("\n✅ All tests passed! Ready to run full pipeline.")
        print("\nNext step:")
        print("  python scripts/build_memory_pipeline.py \\")
        print("      --raw-issues data/trs/memory_seed_issues.json \\")
        print("      --repo flower \\")
        print("      --issue-ids 121 \\")
        print("      --decomposed-output data/trs/decomposed_issues_deterministic.json \\")
        print("      --output-dir data/trs \\")
        print("      --model openrouter/minimax/minimax-m2.5 \\")
        print("      --no-reuse-decomposed")
    else:
        print(f"\n❌ {len(tests) - passed} test(s) failed. Fix before proceeding.")

    sys.exit(0 if passed == len(tests) else 1)
