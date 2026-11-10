"""
Attribution Precision & Recall Evaluation Framework
For evaluating PR generation systems with LLM-assisted + Human verification
"""

import json
import difflib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Change:
    """Represents a single code change"""
    file_path: str
    start_line: int
    end_line: int
    change_type: str  # 'add', 'delete', 'modify'
    content: str

    def to_key(self) -> Tuple:
        """Create hashable key for set operations"""
        return (self.file_path, self.start_line, self.end_line, self.change_type)

    def __hash__(self):
        return hash(self.to_key())

    def __eq__(self, other):
        return self.to_key() == other.to_key()


@dataclass
class PRAnnotation:
    """Annotation for one PR evaluation"""
    pr_id: str
    issue_description: str

    # Ground truth changes (what SHOULD be changed)
    ground_truth_changes: List[Change]

    # System predicted changes (what system ACTUALLY changed)
    system_changes: List[Change]

    # Metadata
    annotator: str  # 'llm', 'human', 'llm+human'
    confidence: float  # 0-1, how confident in this annotation
    notes: str = ""


@dataclass
class EvaluationResult:
    """Results for one PR"""
    pr_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    # Detailed breakdown
    tp_changes: List[Change]
    fp_changes: List[Change]
    fn_changes: List[Change]


class AttributionEvaluator:
    """Evaluates attribution precision and recall"""

    def __init__(self, fuzzy_match: bool = True, line_tolerance: int = 2):
        """
        Args:
            fuzzy_match: Allow approximate matching of line ranges
            line_tolerance: Lines of tolerance for fuzzy matching
        """
        self.fuzzy_match = fuzzy_match
        self.line_tolerance = line_tolerance

    def changes_match(self, c1: Change, c2: Change) -> bool:
        """Check if two changes are considered the same"""
        if c1.file_path != c2.file_path or c1.change_type != c2.change_type:
            return False

        if not self.fuzzy_match:
            return c1.start_line == c2.start_line and c1.end_line == c2.end_line

        # Fuzzy matching: allow small line number differences
        return (abs(c1.start_line - c2.start_line) <= self.line_tolerance and
                abs(c1.end_line - c2.end_line) <= self.line_tolerance)

    def find_matches(self, changes1: List[Change], changes2: List[Change]) -> Set[Tuple[Change, Change]]:
        """Find matching changes between two lists"""
        matches = set()
        used_in_changes2 = set()

        for c1 in changes1:
            for i, c2 in enumerate(changes2):
                if i in used_in_changes2:
                    continue
                if self.changes_match(c1, c2):
                    matches.add((c1, c2))
                    used_in_changes2.add(i)
                    break

        return matches

    def evaluate_single_pr(self, annotation: PRAnnotation) -> EvaluationResult:
        """Evaluate one PR annotation"""
        gt_changes = annotation.ground_truth_changes
        sp_changes = annotation.system_changes

        # Find matches (True Positives)
        matches = self.find_matches(gt_changes, sp_changes)
        tp_changes = [match[1] for match in matches]  # Use system version

        # False Positives: system made but not in ground truth
        fp_changes = [c for c in sp_changes if not any(self.changes_match(c, gt) for gt in gt_changes)]

        # False Negatives: in ground truth but system missed
        fn_changes = [c for c in gt_changes if not any(self.changes_match(gt, sp) for sp in sp_changes)]

        # Calculate metrics
        tp = len(tp_changes)
        fp = len(fp_changes)
        fn = len(fn_changes)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvaluationResult(
            pr_id=annotation.pr_id,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            tp_changes=tp_changes,
            fp_changes=fp_changes,
            fn_changes=fn_changes
        )

    def evaluate_batch(self, annotations: List[PRAnnotation]) -> Dict:
        """Evaluate multiple PRs and return aggregate metrics"""
        results = [self.evaluate_single_pr(ann) for ann in annotations]

        # Micro-averaged metrics (treat all changes equally)
        total_tp = sum(r.true_positives for r in results)
        total_fp = sum(r.false_positives for r in results)
        total_fn = sum(r.false_negatives for r in results)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Macro-averaged metrics (treat all PRs equally)
        macro_precision = sum(r.precision for r in results) / len(results) if results else 0.0
        macro_recall = sum(r.recall for r in results) / len(results) if results else 0.0
        macro_f1 = sum(r.f1 for r in results) / len(results) if results else 0.0

        return {
            'per_pr_results': results,
            'aggregate': {
                'total_prs': len(results),
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn
            }
        }


class DiffParser:
    """Parse git diffs to extract changes"""

    @staticmethod
    def parse_diff_to_changes(diff_text: str) -> List[Change]:
        """Parse unified diff format to Change objects"""
        changes = []
        current_file = None

        for line in diff_text.split('\n'):
            # File header
            if line.startswith('--- a/'):
                current_file = line[6:]
            elif line.startswith('+++ b/'):
                current_file = line[6:]

            # Hunk header: @@ -start,count +start,count @@
            elif line.startswith('@@'):
                # Extract line numbers
                parts = line.split('@@')[1].strip().split()
                if len(parts) >= 2:
                    # Parse added lines range
                    added_range = parts[1].strip('+').split(',')
                    start_line = int(added_range[0])
                    count = int(added_range[1]) if len(added_range) > 1 else 1

                    # This is simplified - you'd need more logic to determine change_type
                    changes.append(Change(
                        file_path=current_file,
                        start_line=start_line,
                        end_line=start_line + count - 1,
                        change_type='modify',
                        content=line
                    ))

        return changes

    @staticmethod
    def parse_github_pr_diff(pr_files: List[Dict]) -> List[Change]:
        """Parse GitHub PR API response to changes"""
        changes = []

        for file_data in pr_files:
            filename = file_data['filename']
            patch = file_data.get('patch', '')
            status = file_data['status']  # 'added', 'removed', 'modified'

            # Parse the patch
            current_line = 0
            for line in patch.split('\n'):
                if line.startswith('@@'):
                    # Extract start line from hunk header
                    parts = line.split('@@')[1].strip().split()
                    if len(parts) >= 2:
                        added_range = parts[1].strip('+').split(',')
                        current_line = int(added_range[0])

                elif line.startswith('+') and not line.startswith('+++'):
                    changes.append(Change(
                        file_path=filename,
                        start_line=current_line,
                        end_line=current_line,
                        change_type='add' if status == 'added' else 'modify',
                        content=line[1:]
                    ))
                    current_line += 1

                elif line.startswith('-') and not line.startswith('---'):
                    changes.append(Change(
                        file_path=filename,
                        start_line=current_line,
                        end_line=current_line,
                        change_type='delete',
                        content=line[1:]
                    ))
                    # Don't increment line for deletions

                else:  # Context line
                    current_line += 1

        return changes


def create_llm_annotation_prompt(pr_data: Dict) -> str:
    """Create prompt for LLM to generate ground truth annotation"""

    prompt = f"""You are evaluating a code change attribution system.

**Issue Description:**
{pr_data['issue_description']}

**System-Generated Changes (what the system actually modified):**
```
{pr_data['system_diff']}
```

**Task:**
Analyze the issue and the system's changes. Identify:

1. **True Positives (TP)**: Changes the system made that ARE correct and necessary
2. **False Positives (FP)**: Changes the system made that are INCORRECT or unnecessary
3. **False Negatives (FN)**: Changes that SHOULD have been made but the system MISSED

For each change, provide:
- file_path
- start_line, end_line
- change_type: 'add', 'delete', or 'modify'
- content: brief description
- category: 'tp', 'fp', or 'fn'
- reasoning: why you categorized it this way

Output as JSON:
{{
  "true_positives": [...],
  "false_positives": [...],
  "false_negatives": [...],
  "confidence": 0.0-1.0,
  "notes": "any uncertainties or edge cases"
}}

Be critical and thorough. If you're uncertain, note it in "notes" and lower confidence."""

    return prompt


def save_annotations(annotations: List[PRAnnotation], output_path: str):
    """Save annotations to JSON file"""
    data = [asdict(ann) for ann in annotations]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_annotations(input_path: str) -> List[PRAnnotation]:
    """Load annotations from JSON file"""
    with open(input_path, 'r') as f:
        data = json.load(f)

    annotations = []
    for item in data:
        item['ground_truth_changes'] = [Change(**c) for c in item['ground_truth_changes']]
        item['system_changes'] = [Change(**c) for c in item['system_changes']]
        annotations.append(PRAnnotation(**item))

    return annotations


# Example usage
if __name__ == "__main__":
    # Example: Create a sample annotation
    sample_annotation = PRAnnotation(
        pr_id="PR-001",
        issue_description="Fix null pointer exception in user login",
        ground_truth_changes=[
            Change("auth/login.py", 45, 47, "modify", "Add null check"),
            Change("auth/login.py", 52, 52, "add", "Add error logging"),
            Change("tests/test_login.py", 20, 25, "add", "Add test for null case"),
        ],
        system_changes=[
            Change("auth/login.py", 45, 47, "modify", "Add null check"),  # TP
            Change("auth/login.py", 52, 52, "add", "Add error logging"),   # TP
            Change("auth/utils.py", 10, 12, "modify", "Unrelated change"), # FP
        ],
        annotator="llm+human",
        confidence=0.95,
        notes="System missed the test case addition"
    )

    # Evaluate
    evaluator = AttributionEvaluator(fuzzy_match=True, line_tolerance=2)
    result = evaluator.evaluate_single_pr(sample_annotation)

    print(f"\n=== Evaluation Results for {result.pr_id} ===")
    print(f"Precision: {result.precision:.2%}")
    print(f"Recall: {result.recall:.2%}")
    print(f"F1 Score: {result.f1:.2%}")
    print(f"\nTrue Positives: {result.true_positives}")
    print(f"False Positives: {result.false_positives}")
    print(f"False Negatives: {result.false_negatives}")

    # Print detailed breakdown
    print("\n--- False Positives (Unnecessary Changes) ---")
    for change in result.fp_changes:
        print(f"  {change.file_path}:{change.start_line}-{change.end_line} ({change.change_type})")

    print("\n--- False Negatives (Missed Changes) ---")
    for change in result.fn_changes:
        print(f"  {change.file_path}:{change.start_line}-{change.end_line} ({change.change_type})")
