"""
McNemar's Statistical Significance Test for CI-Repair-Bench Results

This script performs McNemar's test to determine if improvements over baseline
are statistically significant. McNemar's test is appropriate for paired binary
outcomes (pass/fail on the same instances).

Usage:
    python mcnemar_significance_test.py

Output:
    - Detailed statistical test results
    - Summary table for paper
    - LaTeX-formatted results
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import chi2


class McNemarTest:
    """McNemar's test for paired binary data"""

    def __init__(self, eval_ids: List[str]):
        self.eval_ids = eval_ids

    def load_results(self, filepath: str) -> Optional[Dict[str, int]]:
        """Load results from JSONL or JSON file and return dict mapping ID to pass/fail"""
        results = {}
        eval_ids_set = set(self.eval_ids)

        filepath = Path(filepath)
        if not filepath.exists():
            print(f"  ⚠️  File not found: {filepath}")
            return None

        try:
            with open(filepath, 'r') as f:
                # Check if it's a JSON array or JSONL
                content = f.read()
                f.seek(0)

                if content.strip().startswith('['):
                    # JSON array format (predictions.json)
                    data_list = json.load(f)
                else:
                    # JSONL format
                    data_list = [json.loads(line.strip()) for line in f if line.strip()]

                for data in data_list:
                    id_str = str(data.get('id'))
                    if id_str in eval_ids_set:
                        # Handle both 'conclusion' field (JSONL) and 'verification_passed' field (JSON)
                        if 'conclusion' in data:
                            results[id_str] = 1 if data.get('conclusion') == 'success' else 0
                        elif 'verification_passed' in data:
                            results[id_str] = 1 if data.get('verification_passed') is True else 0
                        else:
                            # Default to failure if neither field exists
                            results[id_str] = 0
            return results
        except Exception as e:
            print(f"  ⚠️  Error reading {filepath}: {e}")
            return None

    def compute_contingency_table(self, baseline: Dict[str, int],
                                  method: Dict[str, int]) -> Tuple[int, int, int, int]:
        """
        Compute 2x2 contingency table for McNemar's test

        Returns:
            (both_pass, baseline_only, method_only, both_fail)
        """
        both_pass = 0
        baseline_only = 0
        method_only = 0
        both_fail = 0

        for id_str in self.eval_ids:
            b = baseline.get(id_str, 0)
            m = method.get(id_str, 0)

            if b == 1 and m == 1:
                both_pass += 1
            elif b == 1 and m == 0:
                baseline_only += 1
            elif b == 0 and m == 1:
                method_only += 1
            else:
                both_fail += 1

        return both_pass, baseline_only, method_only, both_fail

    def mcnemar_test(self, baseline: Dict[str, int], method: Dict[str, int],
                    name_baseline: str, name_method: str,
                    verbose: bool = True) -> Optional[Tuple[float, float]]:
        """
        Perform McNemar's test

        McNemar's test statistic with continuity correction:
            χ² = (|b - c| - 1)² / (b + c)

        where:
            b = number of instances where only baseline passed
            c = number of instances where only method passed

        Returns:
            (chi_square, p_value) or None if data missing
        """
        if baseline is None or method is None:
            if verbose:
                print(f"\n❌ Skipping: {name_baseline} vs {name_method} (missing data)")
            return None

        # Compute contingency table
        both_pass, baseline_only, method_only, both_fail = \
            self.compute_contingency_table(baseline, method)

        # McNemar's test uses discordant pairs
        b = baseline_only
        c = method_only

        if b + c == 0:
            chi_square = 0
            p_value = 1.0
        else:
            # McNemar's test with continuity correction
            chi_square = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - chi2.cdf(chi_square, df=1)

        # Calculate totals
        baseline_total = baseline_only + both_pass
        method_total = method_only + both_pass
        n = len(self.eval_ids)
        improvement = method_total - baseline_total
        improvement_pp = (improvement / n) * 100

        if verbose:
            print(f"\n{'='*80}")
            print(f"McNemar's Test: {name_baseline} vs {name_method}")
            print(f"{'='*80}")
            print(f"Dataset size: {n} instances")
            print(f"")
            print(f"Results:")
            print(f"  Baseline: {baseline_total}/{n} ({baseline_total/n*100:5.1f}%)")
            print(f"  Method:   {method_total}/{n} ({method_total/n*100:5.1f}%)")
            print(f"  Improvement: {improvement:+d} instances ({improvement_pp:+.1f} percentage points)")
            print(f"")
            print(f"Contingency Table:")
            print(f"  ┌─────────────────┬─────────────┬─────────────┐")
            print(f"  │                 │ Baseline ✓  │ Baseline ✗  │")
            print(f"  ├─────────────────┼─────────────┼─────────────┤")
            print(f"  │ Method ✓        │  {both_pass:5d}      │  {method_only:5d} (c)  │")
            print(f"  │ Method ✗        │  {baseline_only:5d} (b)  │  {both_fail:5d}      │")
            print(f"  └─────────────────┴─────────────┴─────────────┘")
            print(f"")
            print(f"McNemar's Test Calculation:")
            print(f"  χ² = (|{b} - {c}| - 1)² / ({b} + {c})")
            print(f"  χ² = (|{b - c}| - 1)² / {b + c}")
            print(f"  χ² = {chi_square:.4f}")
            print(f"  p-value = {p_value:.6f}")
            print(f"")

            # Interpretation
            if p_value < 0.001:
                sig_level = "✓✓✓ HIGHLY SIGNIFICANT"
                interpretation = "p < 0.001"
            elif p_value < 0.01:
                sig_level = "✓✓ VERY SIGNIFICANT"
                interpretation = "p < 0.01"
            elif p_value < 0.05:
                sig_level = "✓ SIGNIFICANT"
                interpretation = "p < 0.05"
            else:
                sig_level = "✗ NOT SIGNIFICANT"
                interpretation = "p ≥ 0.05"

            print(f"Result: {sig_level} ({interpretation})")

        return chi_square, p_value, baseline_total, method_total, improvement


def main():
    """Run all statistical significance tests"""

    print("="*80)
    print("McNemar's Statistical Significance Test")
    print("CI-Repair-Bench Results")
    print("="*80)

    # Load evaluation IDs
    base_dir = Path("/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based")
    eval_ids_file = base_dir / "data" / "eval_issue_ids.json"

    with open(eval_ids_file, 'r') as f:
        all_eval_ids = json.load(f)

    print(f"\nTotal evaluation instances: {len(all_eval_ids)}")

    # File paths
    results_dir = base_dir / "results"

    file_paths = {
        # Mini-SWE Agent results
        "minimax_baseline": results_dir / "miniswe-agent/baseline_minimax-m2.5/jobs_results_diff.jsonl",
        "minimax_l1_l2_l3": results_dir / "miniswe-agent/bidirectional/l1_l2_l3_minimax-m2.5/L1_L2_L3/jobs_results_diff.jsonl",
        "deepseek_baseline": results_dir / "miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl",
        "deepseek_l1_l2_l3": results_dir / "miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/jobs_results_diff.jsonl",

        # Codex results (MiniMax)
        "codex_baseline": results_dir / "codex/baseline_minimax_minimax-m2_5/predictions.json",
        "codex_forward": results_dir / "codex/forward/l1_l2_l3_minimax_minimax-m2_5/jobs_results_diff.jsonl",
        "codex_backward": results_dir / "codex/backward/l1_l2_l3_minimax_minimax-m2_5/jobs_results_diff.jsonl",
        "codex_bidirectional": results_dir / "codex/bidirectional/l1_l2_l3_minimax_minimax-m2_5/jobs_results_diff.jsonl",
    }

    # Results storage
    test_results = {}

    # ========================================================================
    # Test 1: 200-instance matched subset
    # ========================================================================
    print("\n" + "="*80)
    print("MATCHED SUBSET: First 200 Instances")
    print("="*80)

    eval_200 = all_eval_ids[:200]
    test_200 = McNemarTest(eval_200)

    # Load results
    results_200 = {
        name: test_200.load_results(str(path))
        for name, path in file_paths.items()
    }

    # Run tests
    print("\n" + "#"*80)
    print("# MiniMax-M2.5 + mini-SWE Agent")
    print("#"*80)
    result = test_200.mcnemar_test(
        results_200["minimax_baseline"],
        results_200["minimax_l1_l2_l3"],
        "MiniMax Baseline (No-Experience)",
        "MiniMax L1_L2_L3 (Our Method)"
    )
    if result:
        test_results["minimax_200"] = result

    print("\n" + "#"*80)
    print("# DeepSeek-V4-Flash + mini-SWE Agent")
    print("#"*80)
    result = test_200.mcnemar_test(
        results_200["deepseek_baseline"],
        results_200["deepseek_l1_l2_l3"],
        "DeepSeek Baseline (No-Experience)",
        "DeepSeek L1_L2_L3 (Our Method)"
    )
    if result:
        test_results["deepseek_200"] = result

    # ========================================================================
    # Test 2: Full 408-instance evaluation
    # ========================================================================
    print("\n\n" + "="*80)
    print("FULL EVALUATION: All 408 Instances")
    print("="*80)

    test_all = McNemarTest(all_eval_ids)

    # Load results
    results_all = {
        name: test_all.load_results(str(path))
        for name, path in file_paths.items()
    }

    # Run tests
    print("\n" + "#"*80)
    print("# MiniMax-M2.5 + mini-SWE Agent")
    print("#"*80)
    result = test_all.mcnemar_test(
        results_all["minimax_baseline"],
        results_all["minimax_l1_l2_l3"],
        "MiniMax Baseline (No-Experience)",
        "MiniMax L1_L2_L3 (Our Method)"
    )
    if result:
        test_results["minimax_408"] = result

    print("\n" + "#"*80)
    print("# DeepSeek-V4-Flash + mini-SWE Agent")
    print("#"*80)
    result = test_all.mcnemar_test(
        results_all["deepseek_baseline"],
        results_all["deepseek_l1_l2_l3"],
        "DeepSeek Baseline (No-Experience)",
        "DeepSeek L1_L2_L3 (Our Method)"
    )
    if result:
        test_results["deepseek_408"] = result

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "="*80)
    print("SUMMARY FOR PAPER")
    print("="*80)

    print("\n200-Instance Matched Subset:")
    if "minimax_200" in test_results:
        chi2, p, base, method, imp = test_results["minimax_200"]
        sig = "✓" if p < 0.05 else "✗"
        print(f"  MiniMax:  {base}/200 → {method}/200 ({imp:+d}), p={p:.6f} {sig}")

    if "deepseek_200" in test_results:
        chi2, p, base, method, imp = test_results["deepseek_200"]
        sig = "✓✓✓" if p < 0.001 else ("✓✓" if p < 0.01 else ("✓" if p < 0.05 else "✗"))
        print(f"  DeepSeek: {base}/200 → {method}/200 ({imp:+d}), p={p:.6f} {sig}")

    print("\n408-Instance Full Evaluation:")
    if "minimax_408" in test_results:
        chi2, p, base, method, imp = test_results["minimax_408"]
        sig = "✓✓✓" if p < 0.001 else ("✓✓" if p < 0.01 else ("✓" if p < 0.05 else "✗"))
        print(f"  MiniMax:  {base}/408 → {method}/408 ({imp:+d}), p={p:.6f} {sig}")

    if "deepseek_408" in test_results:
        chi2, p, base, method, imp = test_results["deepseek_408"]
        sig = "✓✓✓" if p < 0.001 else ("✓✓" if p < 0.01 else ("✓" if p < 0.05 else "✗"))
        print(f"  DeepSeek: {base}/408 → {method}/408 ({imp:+d}), p={p:.6f} {sig}")

    print("\n" + "-"*80)
    print("For Table Caption:")
    print("-"*80)
    all_significant = all(p < 0.05 for _, p, _, _, _ in test_results.values())
    all_highly_significant = all(p < 0.001 for _, p, _, _, _ in
                                 [v for k, v in test_results.items() if "408" in k])

    if all_highly_significant:
        print("✓ All improvements on the full evaluation (408 instances) are")
        print("  statistically significant (McNemar's test, p < 0.001).")
    elif all_significant:
        print("✓ All improvements are statistically significant")
        print("  (McNemar's test, p < 0.05).")
    else:
        print("⚠ Some improvements are statistically significant")
        print("  (McNemar's test, p < 0.05).")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
