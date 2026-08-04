#!/usr/bin/env python3
"""
Test what structure memory plugin returns vs what mini-swe-agent expects.
"""

import json
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_plugin import MemoryPlugin

# Mock CI failure
ci_failure = {
    "error_context": ["error: command failed"],
    "failure_signals": ["test failed"],
    "relevant_files": [{"path": "test.py"}],
    "error_types": [{"type": "TestFailure"}],
    "failed_jobs": [{"name": "test"}],
    "workflow_name": "test",
    "workflow_path": ".github/workflows/test.yml"
}

verification = {
    "validation_sequence": []
}

issue_metadata = {
    "task_id": "125",
    "sha_fail": "abc123",
    "repo": "test/repo"
}

# Initialize memory plugin
memory = MemoryPlugin(
    memory_root=Path("data/back_trs"),
    result_dir="test_output",
    ablation="L1+L2+L3",
    top_k=3,
    enabled=True
)

# Retrieve
print("=" * 80)
print("TESTING MEMORY PLUGIN OUTPUT STRUCTURE")
print("=" * 80)

result = memory.retrieve(
    ci_failure=ci_failure,
    verification=verification,
    issue_metadata=issue_metadata
)

print("\n📦 MEMORY PLUGIN RETURNS:")
print(f"Type: {type(result)}")
print(f"\nTop-level keys: {list(result.keys())}")

# Check what's in problems
if 'problems' in result:
    problems = result['problems']
    print(f"\n✓ problems: {type(problems)} with {len(problems)} items")

    if problems:
        print(f"\n📋 First problem structure:")
        first = problems[0]
        print(f"  Keys: {list(first.keys())}")

        # Check critical fields
        critical_fields = ['problem', 'root_cause', 'files', 'failure_signals', 'repair_strategy']
        for field in critical_fields:
            if field in first:
                value = first[field]
                if field == 'repair_strategy':
                    print(f"  ✓ {field}: {type(value)} with keys {list(value.keys()) if isinstance(value, dict) else 'N/A'}")
                elif isinstance(value, list):
                    print(f"  ✓ {field}: list with {len(value)} items")
                else:
                    print(f"  ✓ {field}: {type(value).__name__}")
            else:
                print(f"  ✗ {field}: MISSING")

# Check metadata
if 'metadata' in result:
    metadata = result['metadata']
    print(f"\n✓ metadata: {type(metadata)}")
    print(f"  Keys: {list(metadata.keys())}")

# Show full structure (truncated)
print("\n" + "=" * 80)
print("FULL STRUCTURE (JSON):")
print("=" * 80)
print(json.dumps(result, indent=2, default=str)[:2000])
print("\n... (truncated)")

print("\n" + "=" * 80)
print("✓ Test complete!")
print("=" * 80)
