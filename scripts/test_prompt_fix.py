#!/usr/bin/env python3
"""
Test that the improved prompts correctly extract validation_order.

This tests the key fix: LLM should return validation_order as a NUMBER, not null.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def test_single_chunk():
    """Test classification on one chunk to verify validation_order is extracted."""
    print("="*80)
    print("TEST: Validation Order Extraction")
    print("="*80)

    # Load issue 121
    issues_path = PROJECT_ROOT / "data/trs/memory_seed_issues.json"
    with open(issues_path) as f:
        all_issues = json.load(f)

    issue = next((i for i in all_issues if str(i.get('id')) == '121'), None)
    if not issue:
        print("❌ Issue 121 not found!")
        return False

    # Get diff
    diff = issue.get('diff', '')

    # Parse diff
    from deterministic_diff_parser import (
        parse_diff_to_structured,
        chunk_structured_diff,
        format_structured_for_llm
    )

    structured = parse_diff_to_structured(diff)
    chunks = chunk_structured_diff(structured, max_files_per_chunk=10)

    # Get first chunk
    chunk = chunks[0]
    formatted = format_structured_for_llm(chunk)

    print(f"\nChunk 1 has {chunk['total_files']} files:")
    for f in chunk['files'][:3]:
        print(f"  - {f['path']} ({f['file_type']})")

    # Get validation sequence
    from minisweagent.run.benchmarks.utils.ci_workflow_aware_retrieval import (
        analyze_workflow_from_benchmark
    )

    print("\nGetting validation sequence...")
    workflow = analyze_workflow_from_benchmark(issue, None, None)
    validation_sequence = workflow.get("validation_sequence", [])

    print(f"\nValidation sequence has {len(validation_sequence)} validations")
    print("\nSample validations:")
    for v in validation_sequence:
        print(f"  Order {v.get('order')}: {v.get('validates')} - {v.get('validation_cmd', '')[:60]}")

    # Initialize LLM
    from decompose_ci_failure import LitellmModel
    llm = LitellmModel("openrouter/minimax/minimax-m2.5")

    # Create a simplified test prompt
    prompt = f"""Map files to their validation commands.

Files in chunk:
{formatted}

Validation sequence (copy "order" numbers from here):
{json.dumps(validation_sequence[:10], indent=2)}

Output JSON with validation_order as NUMBER (not null):
{{
  "validations": [
    {{
      "validation_order": <NUMBER from validation sequence>,
      "validation_cmd": "<command>",
      "files": ["file.py"]
    }}
  ]
}}

CRITICAL: validation_order MUST be a NUMBER, never null.
"""

    print("\n" + "="*80)
    print("Sending test prompt to LLM...")
    print("="*80)

    try:
        from decompose_ci_failure import _invoke_json
        result = _invoke_json(llm, prompt)

        print("\n✅ LLM Response:")
        print(json.dumps(result, indent=2))

        # Check for null validation_order
        validations = result.get("validations", [])
        null_count = 0
        valid_count = 0

        for v in validations:
            order = v.get("validation_order")
            if order is None or order == "null":
                null_count += 1
                print(f"\n❌ FOUND NULL ORDER: {v}")
            else:
                valid_count += 1
                print(f"\n✅ Valid order {order}: {v.get('validation_cmd', '')[:60]}")

        print("\n" + "="*80)
        print(f"Results: {valid_count} valid, {null_count} null")
        print("="*80)

        if null_count == 0 and valid_count > 0:
            print("\n✅ SUCCESS: All validations have proper order numbers!")
            return True
        elif null_count > 0:
            print(f"\n❌ FAILURE: {null_count} validation(s) have null order")
            return False
        else:
            print("\n⚠️  WARNING: No validations found")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_single_chunk()
    sys.exit(0 if success else 1)
