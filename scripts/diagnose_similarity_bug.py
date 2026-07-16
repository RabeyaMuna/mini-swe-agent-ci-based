#!/usr/bin/env python3
"""
Diagnose the similarity score bug.

This script traces through memory retrieval to find where scores are lost.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def check_memory_records_have_scores():
    """Check if stored memory records have similarity_score field."""
    print("=" * 80)
    print("CHECK 1: Do stored memory records have similarity_score?")
    print("=" * 80)

    memory_file = PROJECT_ROOT / "data" / "trs" / "cross_memory.json"

    if not memory_file.exists():
        print(f"[FAIL] {memory_file} not found")
        return

    with open(memory_file) as f:
        memory = json.load(f)

    for level in ['l1', 'l2', 'l3']:
        if level in memory and memory[level]:
            records = memory[level]
            print(f"\n{level.upper()}: {len(records)} records")

            # Check first record
            first = records[0]
            if 'similarity_score' in first:
                print(f"  [OK] HAS similarity_score: {first['similarity_score']}")
            else:
                print(f"  [FAIL] NO similarity_score field!")
                print(f"  Keys: {list(first.keys())[:15]}")

            break  # Just check L1


def simulate_retrieval():
    """Simulate retrieval to see where scores are lost."""
    print("\n" + "=" * 80)
    print("CHECK 2: Simulate retrieval flow")
    print("=" * 80)

    from minisweagent.run.benchmarks.utils.ci_memory_system import CIMemorySystem

    # Create memory system
    memory_root = str(PROJECT_ROOT / "data" / "trs")

    try:
        system = CIMemorySystem.create(
            memory_root=memory_root,
            memory_enabled=True,
            memory_top_k=10
        )

        if not system.is_enabled():
            print("[FAIL] Memory system not enabled")
            return

        print("[OK] Memory system created")

        # Create a simple query
        log_analysis = {
            "error_types": [{"category": "Code Formatting"}],
            "failed_job": [{"command": "black check"}],
            "overall_error_types": ["Code Formatting"]
        }

        instance = {
            "repo_owner": "test",
            "repo_name": "test",
            "workflow_path": ".github/workflows/test.yml"
        }

        print("\nRetrieving memory...")
        result = system.build_and_retrieve(
            log_analysis_result=log_analysis,
            instance=instance
        )

        print(f"Enabled: {result.get('enabled', False)}")
        print(f"Weighted similarity: {result.get('weighted_similarity', 0):.4f}")

        # Check L1 matches
        l1_matches = result.get('l1_matches', [])
        print(f"\nL1 matches: {len(l1_matches)}")

        if l1_matches:
            for i, record in enumerate(l1_matches[:3]):
                score = record.get('similarity_score', 'MISSING')
                print(f"  L1 record {i}: similarity_score = {score}")

        # Check LLM selection
        llm_sel = result.get('llm_selection', {})
        print(f"\nLLM1 used memory: {llm_sel.get('use_memory', False)}")

        candidates = llm_sel.get('relevant_candidates', [])
        print(f"Relevant candidates: {len(candidates)}")

        if candidates:
            for i, cand in enumerate(candidates[:3]):
                score = cand.get('similarity_score', 'MISSING')
                print(f"  Candidate {i}: similarity_score = {score}")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()


def check_debug_logs():
    """Check debug logs for similarity patterns."""
    print("\n" + "=" * 80)
    print("CHECK 3: Analyze debug logs")
    print("=" * 80)

    log_file = PROJECT_ROOT / "results" / "L1_L2_L3" / "memory_retrieval_debug.jsonl"

    if not log_file.exists():
        print(f"[FAIL] {log_file} not found")
        return

    print(f"Reading {log_file}...")

    with open(log_file) as f:
        for i, line in enumerate(f):
            if i >= 5:  # Check first 5 entries
                break

            try:
                entry = json.loads(line)

                instance_id = entry.get('instance_id', 'unknown')
                weighted = entry.get('weighted_sim', 0)
                above_thresh = entry.get('above_threshold', False)

                top_matches = entry.get('top_matches', [])

                print(f"\nInstance {instance_id}:")
                print(f"  Weighted sim: {weighted:.4f}")
                print(f"  Above threshold: {above_thresh}")

                if top_matches:
                    top_score = top_matches[0].get('score', 0)
                    print(f"  Top match score: {top_score:.4f}")

                    if weighted > 0.1 and top_score == 0.0:
                        print(f"  [WARN] BUG: Weighted sim {weighted:.4f} but top match is 0.000!")

            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    print("\nDIAGNOSING SIMILARITY SCORE BUG")
    print("=" * 80)

    check_memory_records_have_scores()

    print("\n\n")
    check_debug_logs()

    print("\n\n")
    print("=" * 80)
    print("[CRITICAL] EXPECTED DIAGNOSIS:")
    print("=" * 80)
    print("""
If the bug is:
1. Stored records have NO similarity_score field
   → Fix: Records in cross_memory.json need similarity_score added

2. Retrieved records have similarity_score but it's zeroed later
   → Fix: Find where scores are lost in the pipeline

3. Similarity calculation returns 0.0 for all records
   → Fix: Check embedding similarity calculation

Run this script to identify which case it is.
    """)

    print("\nNEXT STEP:")
    print("After identifying the bug location, check:")
    print("1. memory_plugin.py:1183 - where scores are set")
    print("2. ci_memory_system.py:267-270 - where memories are cleaned")
    print("3. ci_memory_system.py:2703-2719 - where memories are passed to LLM1")
