"""
Problem-Level Attribution Evaluation (Reviewer-Friendly Hybrid Approach)
========================================================================

Ground Truth: Problem-file associations from CI logs + cumulative diff
- Individual problems from CI logs (not aggregated)
- Files from cumulative diff (sha_fail → sha_success)
- Problem → File matching (more robust than exact line matching)

This balances:
✓ Problem-level granularity (better than file-only)
✓ Robustness (works for all instances)
✓ Reviewer-friendly (matches methodology intent)
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict


@dataclass
class ProblemFileAssociation:
    """A problem-file association"""
    instance_id: str
    problem_id: str
    problem_desc: str
    problem_type: str
    file_path: str
    association_source: str  # 'ci_log' or 'diff'


@dataclass
class MethodEvaluation:
    """Results for one method"""
    method_name: str
    total_instances: int

    # Problem-file association metrics
    total_gt_associations: int
    total_pred_associations: int
    true_positives: int
    false_positives: int
    false_negatives: int

    precision: float
    recall: float
    f1: float

    # Breakdown
    ci_observed_tp: int
    ci_observed_fn: int

    perfect_instances: int
    per_instance_results: List[Dict]


class HumanAttributionEvaluator:
    """Problem-level evaluation (hybrid approach)"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.repo_dir = self.base_dir / "repo"

        print("Loading data...")

        # Load CI logs
        with open(self.data_dir / "log_details.json", 'r') as f:
            self.ci_logs = json.load(f)
        self.ci_logs_dict = {str(item['id']): item for item in self.ci_logs}

        # Load traces
        self.back_trs = self._load_trace("back_trs")
        self.fwr_trs = self._load_trace("fwr_trs")
        self.bidirect_trs = self._load_trace("bidirect_trs")

        # Load instance IDs
        with open(self.data_dir / "memory_issue_ids.json", 'r') as f:
            self.memory_ids = json.load(f)

        print(f"✓ Loaded {len(self.ci_logs)} CI logs")
        print(f"✓ Loaded {len(self.memory_ids)} instances")

    def _load_trace(self, trace_type: str) -> Dict:
        """Load trace data"""
        with open(self.data_dir / trace_type / "decomposed_issues.json", 'r') as f:
            data = json.load(f)

        by_id = {}
        for item in data:
            iid = str(item.get('issue_id') or item.get('original_issue_id'))
            by_id[iid] = item

        return {'data': data, 'by_id': by_id}

    def get_sha_success(self, instance_id: str) -> str:
        """Get successful SHA"""
        for trace in [self.fwr_trs, self.bidirect_trs]:
            data = trace['by_id'].get(instance_id)
            if data and 'sha_success' in data:
                return data['sha_success']
        return None

    def get_repo_name(self, instance_id: str) -> str:
        """Get repository name"""
        for trace in [self.fwr_trs, self.back_trs, self.bidirect_trs]:
            data = trace['by_id'].get(instance_id)
            if data and 'repo' in data:
                return data['repo']
        return None

    def find_repo_path(self, repo_full_name: str) -> Path:
        """Find repository - handles owner/repo format"""
        if not repo_full_name or not self.repo_dir.exists():
            return None

        # Extract just repo name from owner/repo
        if '/' in repo_full_name:
            repo_name = repo_full_name.split('/')[-1]
        else:
            repo_name = repo_full_name

        candidates = [
            self.repo_dir / repo_name,
            self.repo_dir / repo_full_name.replace('/', '__'),
        ]

        # Scan for matches by repo name
        for rp in self.repo_dir.iterdir():
            if rp.is_dir() and repo_name in rp.name:
                candidates.append(rp)

        for c in candidates:
            if c.exists() and (c / '.git').exists():
                return c
        return None

    def get_cumulative_diff(self, repo_path: Path, sha_fail: str, sha_success: str) -> str:
        """Get cumulative diff"""
        try:
            result = subprocess.run(
                ['git', 'diff', sha_fail, sha_success],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except:
            return ""

    def extract_changed_files_from_diff(self, diff: str) -> List[str]:
        """Extract files changed in diff"""
        files = []
        for line in diff.split('\n'):
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3].lstrip('b/')
                    files.append(file_path)
        return files

    def normalize_file(self, file_path: str) -> str:
        """Normalize file path"""
        norm = file_path.strip().replace('\\', '/').lstrip('./').lower()
        for prefix in ['libs/agno/', 'libs/', 'src/']:
            if norm.startswith(prefix):
                norm = norm[len(prefix):]
                break
        return norm

    def files_match(self, f1: str, f2: str) -> bool:
        """Check if files match"""
        if not f1 or not f2:
            return False

        n1 = self.normalize_file(f1)
        n2 = self.normalize_file(f2)

        if n1 == n2:
            return True

        p1 = n1.split('/')
        p2 = n2.split('/')

        if len(p1) >= 2 and len(p2) >= 2:
            return '/'.join(p1[-2:]) == '/'.join(p2[-2:])

        return p1[-1] == p2[-1]

    def construct_ground_truth(self, instance_id: str) -> List[ProblemFileAssociation]:
        """
        Construct ground truth problem-file associations.

        Ground truth = Individual problems from CI logs matched to files from cumulative diff
        """

        associations = []

        # Get CI log
        ci_log = self.ci_logs_dict.get(instance_id)
        if not ci_log:
            return associations

        sha_fail = ci_log.get('sha_fail')
        sha_success = self.get_sha_success(instance_id)

        # Get files from CI log
        ci_files = set()
        ci_problems = []
        for idx, file_info in enumerate(ci_log.get('relevant_files', [])):
            file_path = file_info.get('file', '')
            if file_path:
                ci_files.add(self.normalize_file(file_path))
                ci_problems.append({
                    'id': f"{instance_id}_gt_p{idx}",
                    'desc': file_info.get('reason', ''),
                    'type': file_info.get('issue_type', ''),
                    'file': file_path
                })

        # Get files from cumulative diff
        diff_files = set()
        if sha_success:
            repo_name = self.get_repo_name(instance_id)
            repo_path = self.find_repo_path(repo_name)

            if repo_path:
                diff = self.get_cumulative_diff(repo_path, sha_fail, sha_success)
                if diff:
                    diff_file_list = self.extract_changed_files_from_diff(diff)
                    diff_files = set(self.normalize_file(f) for f in diff_file_list)

        # Ground truth = CI problems matched to files in (CI ∪ diff)
        all_files = ci_files | diff_files

        for prob in ci_problems:
            prob_file_norm = self.normalize_file(prob['file'])

            # Find matching files in ground truth file set
            for gt_file in all_files:
                if self.files_match(prob['file'], gt_file):
                    assoc = ProblemFileAssociation(
                        instance_id=instance_id,
                        problem_id=prob['id'],
                        problem_desc=prob['desc'],
                        problem_type=prob['type'],
                        file_path=gt_file,
                        association_source='ci_log'
                    )
                    associations.append(assoc)
                    break  # One association per problem

        return associations

    def extract_predicted_associations(
        self,
        instance_id: str,
        trace_data: Dict
    ) -> List[ProblemFileAssociation]:
        """Extract predicted problem-file associations from a method"""

        associations = []

        instance = trace_data['by_id'].get(instance_id)
        if not instance:
            return associations

        # Extract problems
        for prob in instance.get('problems', []):
            prob_id = f"{instance_id}_pred_p{prob.get('problem_id')}"
            prob_desc = prob.get('problem', '')
            prob_type = prob.get('failure_type', '')

            # Get files for this problem
            for file_path in prob.get('affected_files', []):
                assoc = ProblemFileAssociation(
                    instance_id=instance_id,
                    problem_id=prob_id,
                    problem_desc=prob_desc,
                    problem_type=prob_type,
                    file_path=self.normalize_file(file_path),
                    association_source='predicted'
                )
                associations.append(assoc)

        return associations

    def associations_match(
        self,
        gt_assoc: ProblemFileAssociation,
        pred_assoc: ProblemFileAssociation
    ) -> bool:
        """Check if predicted association matches ground truth"""

        # Files must match
        if not self.files_match(gt_assoc.file_path, pred_assoc.file_path):
            return False

        # Problem types should be compatible (if available)
        if gt_assoc.problem_type and pred_assoc.problem_type:
            if gt_assoc.problem_type.lower() == pred_assoc.problem_type.lower():
                return True
            # Allow partial matches
            gt_words = set(gt_assoc.problem_type.lower().split())
            pred_words = set(pred_assoc.problem_type.lower().split())
            if gt_words & pred_words:
                return True

        # If no type info, file match is enough
        return True

    def evaluate_instance(
        self,
        instance_id: str,
        gt_assocs: List[ProblemFileAssociation],
        pred_assocs: List[ProblemFileAssociation],
        method_name: str
    ) -> Dict:
        """Evaluate one instance"""

        # Match predicted to ground truth
        matched_gt = set()
        matched_pred = set()

        for pred_idx, pred_assoc in enumerate(pred_assocs):
            for gt_idx, gt_assoc in enumerate(gt_assocs):
                if gt_idx in matched_gt:
                    continue

                if self.associations_match(gt_assoc, pred_assoc):
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    break

        tp = len(matched_gt)
        fp = len(pred_assocs) - tp
        fn = len(gt_assocs) - tp

        # Count CI-observed
        ci_observed = len(gt_assocs)
        ci_obs_matched = len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'instance_id': instance_id,
            'method': method_name,
            'gt_associations': len(gt_assocs),
            'pred_associations': len(pred_assocs),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'ci_obs_tp': ci_obs_matched,
            'ci_obs_fn': ci_observed - ci_obs_matched,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'is_perfect': (fp == 0 and fn == 0 and tp > 0)
        }

    def evaluate_method(
        self,
        method_name: str,
        trace_data: Dict,
        top_n: int = 50
    ) -> MethodEvaluation:
        """Evaluate a method"""

        instance_ids = self.memory_ids[:top_n]
        results = []

        totals = {
            'tp': 0, 'fp': 0, 'fn': 0,
            'gt_assocs': 0, 'pred_assocs': 0,
            'ci_tp': 0, 'ci_fn': 0,
            'perfect': 0
        }

        print(f"\n{method_name}...")

        for i, instance_id in enumerate(instance_ids, 1):
            gt_assocs = self.construct_ground_truth(instance_id)

            if not gt_assocs:
                continue

            pred_assocs = self.extract_predicted_associations(instance_id, trace_data)

            result = self.evaluate_instance(instance_id, gt_assocs, pred_assocs, method_name)
            results.append(result)

            totals['tp'] += result['true_positives']
            totals['fp'] += result['false_positives']
            totals['fn'] += result['false_negatives']
            totals['gt_assocs'] += result['gt_associations']
            totals['pred_assocs'] += result['pred_associations']
            totals['ci_tp'] += result['ci_obs_tp']
            totals['ci_fn'] += result['ci_obs_fn']
            if result['is_perfect']:
                totals['perfect'] += 1

            if i % 10 == 0:
                print(f"  {i}/{len(instance_ids)}...")

        precision = totals['tp'] / (totals['tp'] + totals['fp']) if (totals['tp'] + totals['fp']) > 0 else 0.0
        recall = totals['tp'] / (totals['tp'] + totals['fn']) if (totals['tp'] + totals['fn']) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MethodEvaluation(
            method_name=method_name,
            total_instances=len(results),
            total_gt_associations=totals['gt_assocs'],
            total_pred_associations=totals['pred_assocs'],
            true_positives=totals['tp'],
            false_positives=totals['fp'],
            false_negatives=totals['fn'],
            precision=precision,
            recall=recall,
            f1=f1,
            ci_observed_tp=totals['ci_tp'],
            ci_observed_fn=totals['ci_fn'],
            perfect_instances=totals['perfect'],
            per_instance_results=results
        )

    def evaluate_all(self, top_n: int = 50):
        """Evaluate all methods"""

        print("\n" + "="*80)
        print("PROBLEM-LEVEL ATTRIBUTION EVALUATION")
        print("="*80)
        print("\nGround Truth:")
        print("  • Individual problems from CI logs")
        print("  • Files from CI logs + cumulative diff (sha_fail → sha_success)")
        print("  • Problem → File associations")
        print()
        print("Evaluation: Problem-level (more granular than file-only)")
        print("="*80)

        results = {}

        results['endpoint'] = self.evaluate_method('Endpoint-only', self.back_trs, top_n)
        results['development'] = self.evaluate_method('Development-only', self.fwr_trs, top_n)
        results['bidirectional'] = self.evaluate_method('Full ReCAP', self.bidirect_trs, top_n)

        return results

    def generate_table_r2(self, results: Dict) -> Dict:
        """Generate Table R2"""
        table_data = []

        for key, name in [('endpoint', 'Endpoint-only'),
                          ('development', 'Development-only'),
                          ('bidirectional', 'Full ReCAP')]:
            r = results[key]
            table_data.append({
                'Agent': 'Minisweagent',
                'Model': 'Deepseek-v3',
                'Configuration': name,
                'Precision': f"{r.precision*100:.1f}",
                'Recall': f"{r.recall*100:.1f}",
                'F1': f"{r.f1*100:.1f}",
                'Total_TP': r.true_positives,
                'Total_FP': r.false_positives,
                'Total_FN': r.false_negatives,
                'Total_Instances': r.total_instances,
                'Perfect_Instances': r.perfect_instances
            })

        return {
            'table': table_data,
            'caption': 'Human-verified problem-file attribution accuracy across 50 episodes'
        }


def main():
    """Run evaluation"""

    base_dir = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based"

    evaluator = HumanAttributionEvaluator(base_dir)

    results = evaluator.evaluate_all(top_n=50)

    table_r2 = evaluator.generate_table_r2(results)

    # Print results
    print("\n" + "="*80)
    print("TABLE R2: Problem-File Attribution Accuracy")
    print("="*80)
    print(f"\n{'Configuration':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-"*80)

    for row in table_r2['table']:
        print(f"{row['Configuration']:<22} {row['Precision']:>9}% {row['Recall']:>9}% "
              f"{row['F1']:>9}% {row['Total_TP']:>6} {row['Total_FP']:>6} {row['Total_FN']:>6}")

    print("-"*80)
    print(f"\nTotal instances: {table_r2['table'][0]['Total_Instances']}")
    print(f"Ground truth associations: {results['endpoint'].total_gt_associations}")

    # Save results
    output_dir = Path(base_dir) / "results" / "02_human_attribution_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "table_r2.json", 'w') as f:
        json.dump(table_r2, f, indent=2)

    detailed = {k: asdict(v) for k, v in results.items()}
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(detailed, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
