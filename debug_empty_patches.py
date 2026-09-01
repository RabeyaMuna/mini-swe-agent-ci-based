#!/usr/bin/env python3
"""
Debug script to understand why patches are empty.
Tests instance 43 which should generate a patch but returns empty.
"""

import json
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from baseline_experepair.wrapper import generate_patch_experepair_baseline

# Load instance 43 data
with open('data/eval_set.jsonl') as f:
    instances = [json.loads(line) for line in f if line.strip()]

# Find instance 43
instance = None
for inst in instances:
    inst_id = inst.get('instance_id') or inst.get('id') or inst['sha_fail']
    if inst_id == '43':
        instance = inst
        break

if not instance:
    print("ERROR: Instance 43 not found")
    sys.exit(1)

print(f"Testing instance 43: {instance['repo_owner']}/{instance['repo_name']}")
print(f"SHA: {instance['sha_fail']}")
print()

# Prepare test data
issue_description = '\n'.join(
    f"[{step.get('step_name', 'unknown')}]\n{step.get('log', '')}"
    for step in instance.get('logs', [])
) if isinstance(instance.get('logs'), list) else instance.get('logs', '')

changed_files = instance.get('changed_files', [])
repo_path = f"/tmp/experepair_repos/{instance['repo_owner']}_{instance['repo_name']}"

print("Issue description length:", len(issue_description))
print("Changed files:", changed_files)
print()

# Run with detailed logging
print("=" * 80)
print("CALLING EXPEREPAIR WRAPPER")
print("=" * 80)

try:
    result = generate_patch_experepair_baseline(
        issue_description=issue_description,
        changed_files=changed_files,
        repo_path=repo_path,
        model="deepseek-v4-flash",
        diff=instance.get('diff', ''),
        workflow=instance.get('workflow', ''),
        validation_commands=instance.get('validation_commands', ''),
        memory_context={},  # No memory for baseline
        sha_fail=instance['sha_fail'],
        instance_id='43'
    )

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Patch length: {len(result.get('patch', ''))}")
    print(f"Applicable: {result.get('applicable', False)}")
    print(f"Cost: ${result.get('cost', 0):.4f}")

    if result.get('patch'):
        print()
        print("PATCH PREVIEW (first 500 chars):")
        print(result['patch'][:500])
    else:
        print()
        print("⚠️  EMPTY PATCH!")
        print("This is the bug we're investigating.")

    if result.get('error'):
        print()
        print(f"ERROR: {result['error']}")

except Exception as e:
    print(f"EXCEPTION: {e}")
    import traceback
    traceback.print_exc()
