"""
compare_runs.py
===============
Systematically compare baseline vs memory-augmented run outcomes
to find the cases that matter for manual investigation.

Usage:
    python scripts/compare_runs.py \
        --baseline  results/baselines/preds.json \
        --memory    results/l1_l2_l3/preds.json \
        --debug-log results/l1_l2_l3/memory_retrieval_debug.jsonl \
        --traj-dir  results/l1_l2_l3 \
        --output    results/comparison_report.json

What it produces:
    1. Summary: how many in each category (HELPED / HURT / BOTH_FIXED / BOTH_FAILED)
    2. Per-instance breakdown for HELPED and HURT cases
    3. For each: memory retrieval info + key agent steps from trajectory
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_preds(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_debug_log(path: str) -> Dict[str, Any]:
    """Returns dict keyed by instance_id."""
    index = {}
    if not os.path.exists(path):
        return index
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                index[r.get("instance_id", "")] = r
            except json.JSONDecodeError:
                pass
    return index


def load_trajectory(traj_dir: str, instance_id: str, sha_fail: str) -> Optional[Dict]:
    """Try to load the trajectory JSON for an instance."""
    for subdir in [sha_fail, instance_id]:
        for fname in [f"{subdir}.traj.json", "traj.json"]:
            path = os.path.join(traj_dir, subdir, fname)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
    return None


def has_patch(pred: Dict) -> bool:
    return bool((pred or {}).get("diff", "").strip())


# ── Trajectory analysis ───────────────────────────────────────────────────────

def extract_agent_summary(traj: Optional[Dict]) -> Dict[str, Any]:
    """Pull key facts from a trajectory for quick manual review."""
    if not traj:
        return {"available": False}

    info = traj.get("info", {})
    messages = traj.get("messages", [])

    # Count bash commands executed
    bash_calls = sum(
        1 for m in messages
        if m.get("role") == "assistant"
        for action in (m.get("extra", {}).get("actions") or [])
    )

    # Extract the last few bash commands (what the agent did at the end)
    last_commands = []
    for m in reversed(messages):
        if m.get("role") == "assistant":
            for action in reversed(m.get("extra", {}).get("actions") or []):
                cmd = action.get("command", "")
                if cmd:
                    last_commands.append(cmd[:200])
                if len(last_commands) >= 5:
                    break
        if len(last_commands) >= 5:
            break
    last_commands.reverse()

    # Did agent mention memory context in its reasoning?
    # Check first assistant message for memory-related keywords
    memory_referenced = False
    for m in messages:
        if m.get("role") == "assistant":
            content = str(m.get("content") or "")
            if any(kw in content.lower() for kw in ["memory context", "similar past", "fix_direction", "key_insight", "past failure"]):
                memory_referenced = True
            break

    return {
        "available":         True,
        "exit_status":       info.get("exit_status", ""),
        "api_calls":         info.get("model_stats", {}).get("api_calls", 0),
        "cost":              round(float(info.get("model_stats", {}).get("instance_cost") or 0), 4),
        "bash_commands":     bash_calls,
        "memory_referenced": memory_referenced,
        "last_commands":     last_commands,
        "has_patch":         bool(info.get("submission", "").strip()),
    }


# ── Main comparison ───────────────────────────────────────────────────────────

def compare(
    baseline_preds: Dict,
    memory_preds:   Dict,
    debug_log:      Dict,
    traj_dir:       str,
) -> Dict[str, Any]:

    all_ids = set(baseline_preds) | set(memory_preds)

    categories = {
        "HELPED":      [],   # baseline failed, memory succeeded
        "HURT":        [],   # baseline succeeded, memory failed
        "BOTH_FIXED":  [],   # both succeeded
        "BOTH_FAILED": [],   # both failed
    }

    for iid in sorted(all_ids):
        b_pred = baseline_preds.get(iid, {})
        m_pred = memory_preds.get(iid, {})
        b_ok   = has_patch(b_pred)
        m_ok   = has_patch(m_pred)

        if   not b_ok and m_ok:     cat = "HELPED"
        elif b_ok and not m_ok:     cat = "HURT"
        elif b_ok and m_ok:         cat = "BOTH_FIXED"
        else:                       cat = "BOTH_FAILED"

        sha_fail = str(m_pred.get("sha_fail") or b_pred.get("sha_fail") or "")
        mem_info = debug_log.get(iid, {})
        traj     = load_trajectory(traj_dir, iid, sha_fail)
        traj_sum = extract_agent_summary(traj)

        record = {
            "instance_id":        iid,
            "sha_fail":           sha_fail[:12],
            "repo":               mem_info.get("repo", ""),
            "category":           cat,
            # Patch status
            "baseline_has_patch": b_ok,
            "memory_has_patch":   m_ok,
            # Memory retrieval
            "memory_enabled":     mem_info.get("memory_enabled", False),
            "weighted_sim":       mem_info.get("weighted_sim", 0.0),
            "above_threshold":    mem_info.get("above_threshold", False),
            "llm_used_memory":    mem_info.get("llm_used_memory", False),
            "counts":             mem_info.get("counts", {}),
            "level_scores":       mem_info.get("level_scores", {}),
            "error_type":         mem_info.get("error_type", ""),
            "failure_pattern":    mem_info.get("failure_pattern", ""),
            "llm_summary":        mem_info.get("llm_summary", ""),
            "selected_items":     mem_info.get("selected_items", []),
            "top_matches":        mem_info.get("top_matches", []),
            # Agent behavior
            "agent":              traj_sum,
        }
        categories[cat].append(record)

    return categories


# ── Display ───────────────────────────────────────────────────────────────────

def print_summary(categories: Dict):
    total = sum(len(v) for v in categories.values())
    print(f"\n{'='*70}")
    print(f"  RUN COMPARISON SUMMARY  ({total} instances)")
    print(f"{'='*70}")
    for cat, records in categories.items():
        label = {
            "HELPED":      "Memory HELPED  (baseline failed, memory succeeded)",
            "HURT":        "Memory HURT    (baseline succeeded, memory failed) ← investigate",
            "BOTH_FIXED":  "Both succeeded",
            "BOTH_FAILED": "Both failed",
        }[cat]
        print(f"  {len(records):3d}  {label}")


def print_category(records: List[Dict], title: str, show_matches: bool = True, limit: int = 20):
    if not records:
        print(f"\n  (none in {title})")
        return

    print(f"\n{'='*70}")
    print(f"  {title}  ({len(records)} instances, showing {min(limit,len(records))})")
    print(f"{'='*70}")

    for r in records[:limit]:
        print(f"\n  {'─'*60}")
        print(f"  {r['instance_id']}")
        print(f"  Repo: {r['repo']}   sha: {r['sha_fail']}")
        print(f"  Error: {r['error_type']}  /  {r['failure_pattern']}")

        # Memory retrieval
        ws   = r.get("weighted_sim", 0.0)
        thr  = 0.33  # default
        used = r.get("llm_used_memory", False)
        cnts = r.get("counts", {})
        print(f"\n  Memory:  weighted={ws:.3f}  above_threshold={r.get('above_threshold')}  "
              f"llm_used={used}  L1={cnts.get('L1',0)} L2={cnts.get('L2',0)} L3={cnts.get('L3',0)}")

        sel = r.get("selected_items", [])
        if sel:
            print(f"  LLM selected {len(sel)} item(s):")
            for item in sel[:2]:
                print(f"    [{item.get('level','')}] score={item.get('score',0):.3f}  {item.get('key_insight','')[:120]}")
                print(f"         fix: {item.get('fix_direction','')[:120]}")
        else:
            print(f"  LLM selected: none")

        summary_text = r.get("llm_summary", "")
        if summary_text:
            print(f"  LLM summary: {summary_text[:200]}")

        # Agent behavior
        ag = r.get("agent", {})
        if ag.get("available"):
            print(f"\n  Agent:   exit={ag.get('exit_status','')}  calls={ag.get('api_calls',0)}  "
                  f"cost=${ag.get('cost',0):.3f}  memory_referenced={ag.get('memory_referenced')}")
            cmds = ag.get("last_commands", [])
            if cmds:
                print(f"  Last commands:")
                for cmd in cmds:
                    print(f"    $ {cmd[:120]}")

        # Top matches for manual similarity check
        if show_matches:
            top = r.get("top_matches", [])
            if top:
                print(f"\n  Top matches (manual check — are these actually similar?):")
                for m in top[:3]:
                    print(f"    [{m.get('level','')}] score={m.get('score',0):.3f}  file={m.get('file','')}")
                    print(f"      error: {m.get('error_type','')}  pattern: {m.get('failure_pattern','')}")
                    print(f"      reason: {m.get('failure_reason','')[:120]}")
                    print(f"      fix:    {m.get('fix_direction','')[:120]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",  required=True,  help="Baseline preds.json")
    parser.add_argument("--memory",    required=True,  help="Memory-augmented preds.json")
    parser.add_argument("--debug-log", default="",     help="memory_retrieval_debug.jsonl")
    parser.add_argument("--traj-dir",  default="",     help="Directory containing trajectory folders")
    parser.add_argument("--output",    default="",     help="Save full report to JSON")
    parser.add_argument("--show-matches", action="store_true", help="Show raw top matches")
    parser.add_argument("--limit",     type=int, default=20)
    args = parser.parse_args()

    baseline_preds = load_preds(args.baseline)
    memory_preds   = load_preds(args.memory)
    debug_log      = load_debug_log(args.debug_log) if args.debug_log else {}

    print(f"Baseline: {len(baseline_preds)} instances  ({sum(1 for p in baseline_preds.values() if has_patch(p))} with patches)")
    print(f"Memory:   {len(memory_preds)} instances  ({sum(1 for p in memory_preds.values() if has_patch(p))} with patches)")

    categories = compare(baseline_preds, memory_preds, debug_log, args.traj_dir)

    print_summary(categories)
    print_category(categories["HELPED"],      "MEMORY HELPED",      args.show_matches, args.limit)
    print_category(categories["HURT"],        "MEMORY HURT ← key",  args.show_matches, args.limit)
    print_category(categories["BOTH_FAILED"], "BOTH FAILED",         args.show_matches, limit=5)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(categories, f, indent=2, ensure_ascii=False)
        print(f"\nFull report saved to {args.output}")

    # Key diagnostic questions
    print(f"\n{'='*70}")
    print(f"  KEY DIAGNOSTIC QUESTIONS")
    print(f"{'='*70}")

    helped = categories["HELPED"]
    hurt   = categories["HURT"]

    # Q1: In HELPED cases — did memory actually fire?
    helped_mem_used = [r for r in helped if r.get("llm_used_memory")]
    print(f"\n  HELPED cases where memory actually fired: {len(helped_mem_used)}/{len(helped)}")
    print(f"  → If low: memory helped by chance (log analysis / workflow context did the work)")

    # Q2: In HURT cases — did memory inject wrong context?
    hurt_mem_used = [r for r in hurt if r.get("llm_used_memory")]
    print(f"\n  HURT cases where memory fired: {len(hurt_mem_used)}/{len(hurt)}")
    print(f"  → If high: memory is actively misleading the agent")

    # Q3: In BOTH_FAILED cases — did memory fire and still fail?
    both_failed_mem = [r for r in categories["BOTH_FAILED"] if r.get("llm_used_memory")]
    print(f"\n  BOTH_FAILED cases where memory fired but still failed: {len(both_failed_mem)}/{len(categories['BOTH_FAILED'])}")
    print(f"  → Shows memory injection quality — high here means retrieval is not useful enough")

    # Q4: Cases where memory did NOT fire (above_threshold=False) — what error types?
    from collections import Counter
    no_fire = [r for r in categories["BOTH_FAILED"] if not r.get("above_threshold")]
    et_gaps = Counter(r.get("error_type","") for r in no_fire)
    print(f"\n  Top error_types with no memory match (gaps in memory bank):")
    for et, n in et_gaps.most_common(5):
        print(f"    {n:3d}  {et or '(none)'}")

    print()


if __name__ == "__main__":
    main()
