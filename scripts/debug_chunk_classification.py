#!/usr/bin/env python3
"""Debug script to see what LLM returns for chunk classification."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from decompose_ci_failure import (
    LitellmModel,
    _invoke_json,
)

def main():
    # Load issue 121
    issues_path = PROJECT_ROOT / "data/trs/memory_seed_issues.json"
    with open(issues_path) as f:
        all_issues = json.load(f)

    issue = next((i for i in all_issues if str(i.get('id')) == '121'), None)
    if not issue:
        print("Issue 121 not found!")
        return 1

    print(f"Issue 121: {issue.get('repo_name')} - {len(issue.get('diff', ''))} char diff")

    # Get a small chunk of diff
    diff = issue.get('diff', '')
    chunk = diff[:5000]  # First 5000 chars

    print(f"\nChunk (first 500 chars):\n{chunk[:500]}\n")

    # Initialize LLM
    llm = LitellmModel("openrouter/minimax/minimax-m2.5")

    # Simple test prompt
    test_prompt = f"""Analyze this git diff chunk and return valid JSON.

DIFF CHUNK:
{chunk}

Return this JSON structure:
{{
  "files_changed": ["list of file paths"],
  "change_types": ["formatting", "typing", "etc"]
}}

Output ONLY valid JSON - nothing else."""

    print("Sending prompt to LLM...")
    try:
        result = _invoke_json(llm, test_prompt)
        print(f"\n✅ Result type: {type(result)}")
        print(f"✅ Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return 0

if __name__ == "__main__":
    sys.exit(main())
