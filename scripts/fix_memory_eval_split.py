#!/usr/bin/env python3
"""
Fix the memory/eval split to properly follow 30/70 rule by repo_name.

Rules:
1. Group by repo_name only (ignore repo_owner)
2. Sort chronologically (old to new by commit timestamp)
3. First 30% → memory_set.jsonl
4. Last 70% → eval_set.jsonl
5. Repos with only 1 issue → skip from memory (eval only)
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def parse_timestamp(timestamp_str):
    """Parse GitHub timestamp or return None."""
    if not timestamp_str:
        return None
    try:
        # Try ISO format
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except:
        return None

def main():
    # Read the current eval_set.jsonl (contains ALL issues)
    print(f"Reading {DATA_DIR / 'eval_set.jsonl'}...")
    all_issues = []
    with open(DATA_DIR / 'eval_set.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                all_issues.append(json.loads(line))

    print(f"Loaded {len(all_issues)} total issues")

    # Group by repo_name only
    repo_groups = defaultdict(list)
    for issue in all_issues:
        repo_name = issue.get('repo_name', 'unknown')
        repo_groups[repo_name].append(issue)

    print(f"Found {len(repo_groups)} unique repository names")

    # Sort each repo's issues chronologically
    for repo_name, issues in repo_groups.items():
        # Sort by timestamp (workflow logs timestamp or commit timestamp)
        for issue in issues:
            # Try to get timestamp from logs
            timestamp = None
            logs = issue.get('logs', [])
            if logs and isinstance(logs, list) and len(logs) > 0:
                log_entry = logs[0]
                if isinstance(log_entry, dict):
                    timestamp_str = log_entry.get('timestamp', '')
                    timestamp = parse_timestamp(timestamp_str)

            # Fallback to workflow timestamp or commit timestamp
            if not timestamp:
                for field in ['workflow_timestamp', 'commit_timestamp', 'created_at']:
                    if field in issue:
                        timestamp = parse_timestamp(issue.get(field, ''))
                        if timestamp:
                            break

            issue['_sort_timestamp'] = timestamp or datetime.min

        # Sort oldest to newest
        issues.sort(key=lambda x: x['_sort_timestamp'])

    # Split each repo 30/70
    memory_issues = []
    eval_issues = []

    stats = {
        'total_repos': len(repo_groups),
        'single_issue_repos': 0,
        'multi_issue_repos': 0,
        'memory_count': 0,
        'eval_count': 0,
    }

    print("\n" + "="*80)
    print(f"{'Repo Name':<25} {'Total':>6} {'Memory':>7} {'Eval':>6} {'Split':<20}")
    print("="*80)

    for repo_name in sorted(repo_groups.keys(), key=lambda x: len(repo_groups[x]), reverse=True):
        issues = repo_groups[repo_name]
        total = len(issues)

        if total == 1:
            # Single issue → eval only
            eval_issues.extend(issues)
            stats['single_issue_repos'] += 1
            stats['eval_count'] += 1
            print(f"{repo_name:<25} {total:>6} {0:>7} {1:>6} {'single (eval only)':<20}")
        else:
            # Multi-issue → 30/70 split
            split_point = max(1, int(total * 0.3))  # At least 1 in memory

            memory_part = issues[:split_point]
            eval_part = issues[split_point:]

            memory_issues.extend(memory_part)
            eval_issues.extend(eval_part)

            stats['multi_issue_repos'] += 1
            stats['memory_count'] += len(memory_part)
            stats['eval_count'] += len(eval_part)

            pct = len(memory_part) / total * 100
            print(f"{repo_name:<25} {total:>6} {len(memory_part):>7} {len(eval_part):>6} {pct:>5.1f}% in memory")

    print("="*80)
    print(f"Total: {len(all_issues)} issues")
    print(f"  Memory: {len(memory_issues)} ({len(memory_issues)/len(all_issues)*100:.1f}%)")
    print(f"  Eval:   {len(eval_issues)} ({len(eval_issues)/len(all_issues)*100:.1f}%)")
    print("="*80)

    # Backup old files
    print("\nBacking up old files...")
    for filename in ['memory_set.jsonl', 'eval_set.jsonl', 'memory_issue_ids.json', 'eval_issue_ids.json']:
        old_file = DATA_DIR / filename
        if old_file.exists():
            backup = DATA_DIR / f"{filename}.backup"
            old_file.rename(backup)
            print(f"  {filename} → {filename}.backup")

    # Write new memory_set.jsonl
    print(f"\nWriting {DATA_DIR / 'memory_set.jsonl'}...")
    with open(DATA_DIR / 'memory_set.jsonl', 'w') as f:
        for issue in memory_issues:
            # Remove temporary sort field
            issue.pop('_sort_timestamp', None)
            f.write(json.dumps(issue, ensure_ascii=False) + '\n')

    # Write new eval_set.jsonl
    print(f"Writing {DATA_DIR / 'eval_set.jsonl'}...")
    with open(DATA_DIR / 'eval_set.jsonl', 'w') as f:
        for issue in eval_issues:
            # Remove temporary sort field
            issue.pop('_sort_timestamp', None)
            f.write(json.dumps(issue, ensure_ascii=False) + '\n')

    # Write memory_issue_ids.json
    memory_ids = [issue.get('id') for issue in memory_issues]
    with open(DATA_DIR / 'memory_issue_ids.json', 'w') as f:
        json.dump(memory_ids, f, indent=2)

    # Write eval_issue_ids.json
    eval_ids = [issue.get('id') for issue in eval_issues]
    with open(DATA_DIR / 'eval_issue_ids.json', 'w') as f:
        json.dump(eval_ids, f, indent=2)

    # Update split_metadata.json
    metadata = {
        'total_issues': len(all_issues),
        'memory_size': len(memory_issues),
        'eval_size': len(eval_issues),
        'memory_ratio': len(memory_issues) / len(all_issues),
        'selection_strategy': 'chronological_by_repo',
        'split_by_repo': True,
        'split_by_repo_name_only': True,
        'temporal_leakage_prevented': True,
        'single_issue_repos_in_eval_only': True,
        'repos': {
            'total': stats['total_repos'],
            'single_issue': stats['single_issue_repos'],
            'multi_issue': stats['multi_issue_repos'],
        }
    }

    with open(DATA_DIR / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ Split completed successfully!")
    print(f"\nFiles created:")
    print(f"  - {DATA_DIR / 'memory_set.jsonl'} ({len(memory_issues)} issues)")
    print(f"  - {DATA_DIR / 'eval_set.jsonl'} ({len(eval_issues)} issues)")
    print(f"  - {DATA_DIR / 'memory_issue_ids.json'} ({len(memory_ids)} IDs)")
    print(f"  - {DATA_DIR / 'eval_issue_ids.json'} ({len(eval_ids)} IDs)")
    print(f"  - {DATA_DIR / 'split_metadata.json'} (updated)")

    print(f"\nBackups saved with .backup extension")

if __name__ == '__main__':
    main()
