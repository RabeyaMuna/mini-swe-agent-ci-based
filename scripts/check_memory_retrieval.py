"""
check_memory_retrieval.py
=========================
Read memory_retrieval_debug.jsonl and answer the four key questions:

  1. Was memory retrieved?          (similarity >= threshold?)
  2. How many matches per level?    (L1: X, L2: Y, L3: Z)
  3. What are the similarity scores?
  4. Are the matched issues similar? (manual check — shows the actual records)

Usage:
    python scripts/check_memory_retrieval.py --log results/l1_l2_l3/memory_retrieval_debug.jsonl
    python scripts/check_memory_retrieval.py --log results/l1_l2_l3/memory_retrieval_debug.jsonl --show-matches
    python scripts/check_memory_retrieval.py --log results/l1_l2_l3/memory_retrieval_debug.jsonl --instance agno-agi__agno__abc123
    python scripts/check_memory_retrieval.py --log results/l1_l2_l3/memory_retrieval_debug.jsonl --only-used
    python scripts/check_memory_retrieval.py --log results/l1_l2_l3/memory_retrieval_debug.jsonl --only-missed
"""

import argparse
import json
import os
from collections import Counter


def load_log(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def summary(records):
    total = len(records)
    enabled   = [r for r in records if r.get("memory_enabled")]
    above     = [r for r in enabled if r.get("above_threshold")]
    llm_used  = [r for r in enabled if r.get("llm_used_memory")]
    no_match  = [r for r in enabled if sum(r.get("counts", {}).values()) == 0]

    print(f"\n{'='*65}")
    print(f"  MEMORY RETRIEVAL SUMMARY  ({total} instances)")
    print(f"{'='*65}")
    print(f"  Memory enabled:           {len(enabled):3d} / {total}")
    print(f"  Above similarity threshold: {len(above):3d} / {len(enabled)}   ({100*len(above)/max(len(enabled),1):.1f}%)")
    print(f"  LLM gate used memory:     {len(llm_used):3d} / {len(enabled)}   ({100*len(llm_used)/max(len(enabled),1):.1f}%)")
    print(f"  Zero matches found:       {len(no_match):3d} / {len(enabled)}")

    # Average scores
    ws = [r.get("weighted_sim", 0.0) for r in enabled]
    l1s = [r.get("level_scores", {}).get("L1", 0.0) for r in enabled]
    l2s = [r.get("level_scores", {}).get("L2", 0.0) for r in enabled]
    l3s = [r.get("level_scores", {}).get("L3", 0.0) for r in enabled]
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    print(f"\n  Average scores (all enabled):")
    print(f"    Weighted: {avg(ws):.3f}  L1: {avg(l1s):.3f}  L2: {avg(l2s):.3f}  L3: {avg(l3s):.3f}")

    print(f"\n  Average match counts (all enabled):")
    l1c = avg([sum(r.get("counts",{}).get("L1",0) for _ in [1]) for r in enabled])
    l2c = avg([sum(r.get("counts",{}).get("L2",0) for _ in [1]) for r in enabled])
    l3c = avg([sum(r.get("counts",{}).get("L3",0) for _ in [1]) for r in enabled])
    print(f"    L1: {l1c:.1f}  L2: {l2c:.1f}  L3: {l3c:.1f}")

    # Top error types that triggered memory use
    et_used = Counter(r.get("error_type","") for r in llm_used)
    print(f"\n  Top error_types where memory was used:")
    for et, n in et_used.most_common(6):
        print(f"    {n:3d}  {et or '(none)'}")

    # Top error types that got zero matches
    et_miss = Counter(r.get("error_type","") for r in no_match)
    if et_miss:
        print(f"\n  Top error_types with zero matches (memory gaps):")
        for et, n in et_miss.most_common(6):
            print(f"    {n:3d}  {et or '(none)'}")


def show_instance(r, show_matches=False):
    print(f"\n{'─'*65}")
    print(f"  Instance: {r.get('instance_id','')}")
    print(f"  Repo:     {r.get('repo','')}   sha: {r.get('sha_fail','')}")
    print(f"  Error:    {r.get('error_type','')}  /  {r.get('failure_pattern','')}")
    print(f"  Reason:   {r.get('failure_reason','')[:120]}")
    print()

    # Q1: Was memory retrieved?
    above = r.get("above_threshold", False)
    ws    = r.get("weighted_sim", 0.0)
    thr   = r.get("threshold", 0.0)
    used  = r.get("llm_used_memory", False)
    print(f"  Q1  Was memory retrieved?     {'YES OK' if above else 'NO ERROR'}  (weighted={ws:.3f}  threshold={thr:.3f})")
    print(f"      LLM gate used memory?     {'YES OK' if used else 'NO ERROR'}")

    # Q2: Match counts
    counts = r.get("counts", {})
    print(f"\n  Q2  Matches found:")
    print(f"      L1={counts.get('L1',0)}  L2={counts.get('L2',0)}  L3={counts.get('L3',0)}")

    # Q3: Scores
    sc = r.get("level_scores", {})
    print(f"\n  Q3  Similarity scores:")
    print(f"      L1={sc.get('L1',0):.3f}  L2={sc.get('L2',0):.3f}  L3={sc.get('L3',0):.3f}  weighted={ws:.3f}")

    # LLM summary
    summary_text = r.get("llm_summary", "")
    if summary_text:
        print(f"\n      LLM summary: {summary_text[:300]}")

    # Selected items
    selected = r.get("selected_items", [])
    if selected:
        print(f"\n  LLM-selected items ({len(selected)}):")
        for item in selected:
            print(f"    [{item.get('level','')}] score={item.get('score',0):.3f}  relevance={item.get('relevance','')}")
            print(f"      insight:   {item.get('key_insight','')[:150]}")
            print(f"      fix:       {item.get('fix_direction','')[:150]}")
    else:
        print(f"\n  LLM-selected items: none")

    # Q4: Are the matched issues actually similar? (raw top matches)
    if show_matches:
        top = r.get("top_matches", [])
        if top:
            print(f"\n  Q4  Top raw matches (manual similarity check):")
            for m in top:
                print(f"    [{m.get('level','')}] score={m.get('score',0):.3f}  file={m.get('file','')}")
                print(f"      error_type:   {m.get('error_type','')}")
                print(f"      pattern:      {m.get('failure_pattern','')}")
                print(f"      reason:       {m.get('failure_reason','')[:150]}")
                print(f"      fix:          {m.get('fix_direction','')[:150]}")
                mo = m.get("matched_on", {})
                if mo:
                    scores_str = "  ".join(f"{k}={v:.3f}" for k, v in mo.items())
                    print(f"      matched_on:   {scores_str}")
        else:
            print(f"\n  Q4  No raw matches to show.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to memory_retrieval_debug.jsonl")
    parser.add_argument("--show-matches", action="store_true", help="Show raw top matches for Q4")
    parser.add_argument("--instance", default="", help="Show details for one instance_id")
    parser.add_argument("--only-used", action="store_true", help="Show only instances where LLM used memory")
    parser.add_argument("--only-missed", action="store_true", help="Show only instances where memory was not retrieved")
    parser.add_argument("--limit", type=int, default=10, help="Max instances to show in detail (default 10)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}")
        return

    records = load_log(args.log)
    if not records:
        print("Log is empty.")
        return

    # Overall summary always shown
    summary(records)

    # Filter for detail view
    if args.instance:
        detail = [r for r in records if args.instance in r.get("instance_id", "")]
    elif args.only_used:
        detail = [r for r in records if r.get("llm_used_memory")]
    elif args.only_missed:
        detail = [r for r in records if not r.get("above_threshold") and r.get("memory_enabled")]
    else:
        detail = records

    if not detail:
        print("\nNo matching records for detail view.")
        return

    print(f"\n{'='*65}")
    label = "all" if not (args.instance or args.only_used or args.only_missed) else ("used" if args.only_used else ("missed" if args.only_missed else args.instance))
    print(f"  DETAIL VIEW  [{label}]  showing {min(args.limit, len(detail))} of {len(detail)}")
    print(f"{'='*65}")

    for r in detail[:args.limit]:
        show_instance(r, show_matches=args.show_matches)

    print()


if __name__ == "__main__":
    main()
