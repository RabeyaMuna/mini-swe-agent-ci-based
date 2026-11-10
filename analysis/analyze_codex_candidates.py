#!/usr/bin/env python3
"""
Analyze candidate issues for Codex improvement.
Compare issues that pass on MiniSWE-DeepSeek but fail on Codex.
"""

import json
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

def load_data():
    """Load all necessary data files."""

    # Load first 200 eval IDs
    with open(PROJECT_ROOT / 'data' / 'eval_issue_ids.json') as f:
        eval_200 = set(json.load(f)[:200])

    # Load Codex CI results
    codex_ci_results = {}
    with open(PROJECT_ROOT / 'results/codex/bidirectional/l1_l2_l3_minimax_minimax-m2_5/jobs_results_diff.jsonl') as f:
        for line in f:
            item = json.loads(line)
            if item['id'] in eval_200:
                codex_ci_results[item['id']] = item

    # Load MiniSWE CI results
    miniswe_ci_results = {}
    with open(PROJECT_ROOT / 'results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/jobs_results_diff.jsonl') as f:
        for line in f:
            item = json.loads(line)
            if item['id'] in eval_200:
                miniswe_ci_results[item['id']] = item

    # Load Codex predictions
    with open(PROJECT_ROOT / 'results/codex/bidirectional/l1_l2_l3_minimax_minimax-m2_5/predictions.json') as f:
        codex_preds_list = json.load(f)

    codex_preds = {item['id']: item for item in codex_preds_list if item['id'] in eval_200}

    # Load MiniSWE predictions
    with open(PROJECT_ROOT / 'results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/preds.json') as f:
        miniswe_preds = json.load(f)

    miniswe_preds = {k: v for k, v in miniswe_preds.items() if k in eval_200}

    return eval_200, codex_ci_results, miniswe_ci_results, codex_preds, miniswe_preds


def find_candidates(codex_ci_results, miniswe_ci_results):
    """Find issues that pass on MiniSWE but fail on Codex."""

    codex_passed = {
        issue_id for issue_id, result in codex_ci_results.items()
        if result.get('conclusion') == 'success' and result.get('patch_applied')
    }

    miniswe_passed = {
        issue_id for issue_id, result in miniswe_ci_results.items()
        if result.get('conclusion') == 'success' and result.get('patch_applied')
    }

    candidates = miniswe_passed - codex_passed

    return sorted(candidates, key=lambda x: int(x))


def analyze_candidate(issue_id, codex_ci_results, miniswe_ci_results, codex_preds, miniswe_preds):
    """Analyze a single candidate issue."""

    print(f"\n{'='*80}")
    print(f"Issue {issue_id}")
    print(f"{'='*80}")

    # Codex info
    if issue_id in codex_ci_results:
        codex_ci = codex_ci_results[issue_id]
        print(f"\n📊 Codex CI Result:")
        print(f"   Repo: {codex_ci.get('repo_name')}")
        print(f"   Workflow: {codex_ci.get('workflow')}")
        print(f"   Conclusion: {codex_ci.get('conclusion')}")
        print(f"   Patch Applied: {codex_ci.get('patch_applied')}")
        print(f"   URL: {codex_ci.get('url')}")

    if issue_id in codex_preds:
        codex_pred = codex_preds[issue_id]
        print(f"\n📊 Codex Prediction:")
        print(f"   Patch Generated: {codex_pred.get('patch_generated')}")
        print(f"   Changed Files: {codex_pred.get('changed_files', [])}")
        print(f"   Cost: ${codex_pred.get('total_cost_usd', 0):.4f}")
        print(f"   Time: {codex_pred.get('total_elapsed_seconds', 0):.1f}s")

    # MiniSWE info
    if issue_id in miniswe_ci_results:
        miniswe_ci = miniswe_ci_results[issue_id]
        print(f"\n✅ MiniSWE-DeepSeek CI Result:")
        print(f"   Repo: {miniswe_ci.get('repo_name')}")
        print(f"   Workflow: {miniswe_ci.get('workflow')}")
        print(f"   Conclusion: {miniswe_ci.get('conclusion')}")
        print(f"   Patch Applied: {miniswe_ci.get('patch_applied')}")
        print(f"   URL: {miniswe_ci.get('url')}")

    if issue_id in miniswe_preds:
        miniswe_pred = miniswe_preds[issue_id]
        print(f"\n✅ MiniSWE-DeepSeek Prediction:")
        print(f"   Changed Files: {miniswe_pred.get('changed_files', [])}")

    # Compare
    print(f"\n🔍 Comparison:")
    if issue_id in codex_preds and issue_id in miniswe_preds:
        codex_files = set(codex_preds[issue_id].get('changed_files', []))
        miniswe_files = set(miniswe_preds[issue_id].get('changed_files', []))

        if codex_files == miniswe_files:
            print(f"   ✓ Same files changed")
        else:
            print(f"   ✗ Different files changed:")
            print(f"     Codex only: {codex_files - miniswe_files}")
            print(f"     MiniSWE only: {miniswe_files - codex_files}")
            print(f"     Both: {codex_files & miniswe_files}")


def main():
    """Main analysis function."""

    print("Loading data...")
    eval_200, codex_ci_results, miniswe_ci_results, codex_preds, miniswe_preds = load_data()

    print("Finding candidates...")
    candidates = find_candidates(codex_ci_results, miniswe_ci_results)

    # Summary
    codex_passed = sum(
        1 for result in codex_ci_results.values()
        if result.get('conclusion') == 'success' and result.get('patch_applied')
    )
    miniswe_passed = sum(
        1 for result in miniswe_ci_results.values()
        if result.get('conclusion') == 'success' and result.get('patch_applied')
    )

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nCodex: {codex_passed} passed")
    print(f"MiniSWE-DeepSeek: {miniswe_passed} passed")
    print(f"Candidates (pass on MiniSWE but not Codex): {len(candidates)}")
    print(f"\nCandidate Issues: {candidates}")

    # Analyze each candidate
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS")
    print(f"{'='*80}")

    for issue_id in candidates:
        analyze_candidate(issue_id, codex_ci_results, miniswe_ci_results, codex_preds, miniswe_preds)

    # Save detailed report
    report = {
        'summary': {
            'codex_passed': codex_passed,
            'miniswe_passed': miniswe_passed,
            'candidates_count': len(candidates),
            'candidates': candidates
        },
        'candidates_detail': []
    }

    for issue_id in candidates:
        detail = {
            'issue_id': issue_id,
            'codex': {
                'ci': codex_ci_results.get(issue_id, {}),
                'pred': codex_preds.get(issue_id, {})
            },
            'miniswe': {
                'ci': miniswe_ci_results.get(issue_id, {}),
                'pred': miniswe_preds.get(issue_id, {})
            }
        }
        report['candidates_detail'].append(detail)

    output_file = PROJECT_ROOT / 'analysis' / 'codex_candidates_detailed.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Detailed report saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
