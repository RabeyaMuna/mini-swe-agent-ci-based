"""
Human Annotation Interface for Problem-Hunk Attribution
Simulates careful human review of each problem-hunk association.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import re


@dataclass
class AnnotatedAssociation:
    """A human-annotated problem-hunk association"""
    instance_id: str
    problem_id: int
    problem_description: str
    problem_type: str
    failure_evidence: str

    file_path: str
    hunk_start: int
    hunk_end: int
    hunk_diff: str

    # Human annotation
    association_label: str  # 'direct', 'shared', 'unrelated', 'uncertain'
    confidence: str  # 'high', 'medium', 'low'
    annotation_rationale: str

    # Predicted by each method
    predicted_by_endpoint: bool
    predicted_by_development: bool
    predicted_by_bidirectional: bool


class HumanAnnotationSimulator:
    """
    Simulates careful human annotation of problem-hunk pairs.
    Uses rule-based heuristics that mimic human reasoning.
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"

        # Load data
        self.log_details = self._load_json(self.data_dir / "log_details.json")
        self.log_details_dict = {str(item['id']): item for item in self.log_details}

    def _load_json(self, path: Path) -> any:
        """Load JSON file"""
        with open(path, 'r') as f:
            return json.load(f)

    def annotate_association(
        self,
        problem: Dict,
        hunk: Dict,
        ci_failure: Dict,
        instance_id: str
    ) -> AnnotatedAssociation:
        """
        Simulate human annotation of a problem-hunk pair.

        A human would consider:
        1. Does the CI error message reference this file/line?
        2. Does the hunk change address the type of error reported?
        3. Is the fix localized to where the error was reported?
        4. Could this be a shared fix (multiple problems)?
        """

        # Extract problem details
        problem_desc = problem.get('problem', '')
        problem_type = problem.get('failure_type', '')
        validation_cmd = problem.get('validation_cmd', '')
        affected_files = problem.get('affected_files', [])

        # Extract hunk details
        file_path = hunk.get('file', '')
        hunk_start = hunk.get('start_line', 0)
        hunk_end = hunk.get('end_line', 0)
        hunk_content = hunk.get('content', '')
        added_lines = hunk.get('added_lines', [])
        removed_lines = hunk.get('removed_lines', [])

        # Get failure evidence
        failure_evidence = self._extract_failure_evidence(
            ci_failure, file_path, hunk_start, hunk_end
        )

        # Annotation logic (simulating human reasoning)
        association_label, confidence, rationale = self._determine_association(
            problem_desc=problem_desc,
            problem_type=problem_type,
            validation_cmd=validation_cmd,
            affected_files=affected_files,
            file_path=file_path,
            hunk_start=hunk_start,
            hunk_end=hunk_end,
            added_lines=added_lines,
            removed_lines=removed_lines,
            failure_evidence=failure_evidence
        )

        return AnnotatedAssociation(
            instance_id=instance_id,
            problem_id=problem.get('problem_id', 0),
            problem_description=problem_desc,
            problem_type=problem_type,
            failure_evidence=failure_evidence,
            file_path=file_path,
            hunk_start=hunk_start,
            hunk_end=hunk_end,
            hunk_diff=hunk_content,
            association_label=association_label,
            confidence=confidence,
            annotation_rationale=rationale,
            predicted_by_endpoint=False,  # Set by caller
            predicted_by_development=False,
            predicted_by_bidirectional=False
        )

    def _extract_failure_evidence(
        self,
        ci_failure: Dict,
        file_path: str,
        hunk_start: int,
        hunk_end: int
    ) -> str:
        """Extract specific failure evidence for this file/line range"""
        evidence_parts = []

        # Check failure signals
        for signal in ci_failure.get('failure_signals', []):
            # Extract file and line from signal
            if file_path.split('/')[-1] in signal:
                evidence_parts.append(signal)

        # Check relevant files
        for file_info in ci_failure.get('relevant_files', []):
            if self._file_matches(file_info.get('file', ''), file_path):
                line_num = file_info.get('line_number')
                if line_num and hunk_start <= line_num <= hunk_end:
                    evidence_parts.append(
                        f"{file_info.get('failed_tool', '')}: {file_info.get('reason', '')}"
                    )

        return " | ".join(evidence_parts) if evidence_parts else "No direct evidence"

    def _determine_association(
        self,
        problem_desc: str,
        problem_type: str,
        validation_cmd: str,
        affected_files: List[str],
        file_path: str,
        hunk_start: int,
        hunk_end: int,
        added_lines: List[str],
        removed_lines: List[str],
        failure_evidence: str
    ) -> Tuple[str, str, str]:
        """
        Determine association type using human-like reasoning.

        Returns: (association_label, confidence, rationale)
        """

        # Check 1: Is this file in affected_files?
        file_is_affected = any(
            self._file_matches(af, file_path) for af in affected_files
        )

        if not file_is_affected:
            return ('unrelated', 'high',
                   'File not listed in affected files for this problem')

        # Check 2: Does the failure evidence directly reference this location?
        has_direct_evidence = failure_evidence != "No direct evidence"

        # Check 3: Analyze the fix pattern
        fix_pattern = self._analyze_fix_pattern(
            problem_desc, problem_type, added_lines, removed_lines
        )

        # Decision logic
        if has_direct_evidence and fix_pattern['matches']:
            # Strong match: evidence points here and fix makes sense
            return ('direct', 'high',
                   f'CI error directly references this location. '
                   f'Fix pattern matches: {fix_pattern["reason"]}')

        elif has_direct_evidence and not fix_pattern['matches']:
            # Evidence points here but fix doesn't match expected pattern
            return ('uncertain', 'medium',
                   f'CI error references this location but fix pattern unclear: '
                   f'{fix_pattern["reason"]}')

        elif not has_direct_evidence and fix_pattern['matches']:
            # No direct evidence but fix pattern matches
            # Could be a shared fix or indirect relationship
            if 'type' in problem_desc.lower() or 'import' in problem_desc.lower():
                # Type/import fixes often affect multiple locations
                return ('shared', 'medium',
                       f'No direct CI reference, but fix pattern matches problem type. '
                       f'Likely shared fix: {fix_pattern["reason"]}')
            else:
                return ('uncertain', 'low',
                       f'No direct CI reference. Fix may be related: {fix_pattern["reason"]}')

        else:
            # No evidence and no pattern match
            return ('unrelated', 'medium',
                   'No direct CI reference and fix pattern does not match problem type')

    def _analyze_fix_pattern(
        self,
        problem_desc: str,
        problem_type: str,
        added_lines: List[str],
        removed_lines: List[str]
    ) -> Dict:
        """
        Analyze if the fix pattern matches the problem type.
        This simulates human pattern recognition.
        """
        problem_lower = problem_desc.lower()
        type_lower = problem_type.lower()

        # Combine added and removed for analysis
        all_changes = '\n'.join(added_lines + removed_lines)

        # Pattern matching rules
        patterns = {
            'type_checking': {
                'keywords': ['type', 'mypy', 'optional', 'union', 'str', 'int', 'none'],
                'fix_indicators': ['Optional', 'Union', 'cast(', '-> ', ': str', ': int',
                                  'isinstance', 'if not None', 'if ... is not None']
            },
            'linting': {
                'keywords': ['lint', 'ruff', 'flake8', 'f-string', 'unused', 'import'],
                'fix_indicators': ['# noqa', 'import ', 'from ', '"""', "'''",
                                  'removed f-string']
            },
            'testing': {
                'keywords': ['test', 'pytest', 'assert', 'mock', 'fixture'],
                'fix_indicators': ['assert', 'mock', '@pytest', 'def test_',
                                  'expected', 'actual']
            },
            'import': {
                'keywords': ['import', 'module', 'cannot import', 'no module'],
                'fix_indicators': ['import ', 'from ', '__init__']
            },
            'syntax': {
                'keywords': ['syntax', 'parsing', 'indentation', 'unexpected'],
                'fix_indicators': ['    ', '\t', ':', '(', ')', '[', ']']
            }
        }

        # Determine problem category
        for category, info in patterns.items():
            if any(kw in problem_lower or kw in type_lower for kw in info['keywords']):
                # Check if fix matches expected pattern
                if any(indicator in all_changes for indicator in info['fix_indicators']):
                    return {
                        'matches': True,
                        'category': category,
                        'reason': f'Fix contains {category} patterns matching the reported error'
                    }

        # Generic check: if problem mentions specific code elements,
        # check if they appear in the diff
        code_elements = re.findall(r'`([^`]+)`', problem_desc)
        if code_elements:
            if any(elem in all_changes for elem in code_elements):
                return {
                    'matches': True,
                    'category': 'specific',
                    'reason': f'Fix modifies mentioned code elements: {code_elements[:2]}'
                }

        return {
            'matches': False,
            'category': 'unknown',
            'reason': 'Fix pattern does not clearly match problem description'
        }

    def _file_matches(self, file1: str, file2: str) -> bool:
        """Check if two file paths refer to the same file"""
        f1 = file1.strip().replace('\\', '/').lstrip('./')
        f2 = file2.strip().replace('\\', '/').lstrip('./')

        if f1 == f2:
            return True

        # Check if filenames match
        f1_parts = f1.split('/')
        f2_parts = f2.split('/')

        if f1_parts[-1] == f2_parts[-1]:
            if len(f1_parts) >= 2 and len(f2_parts) >= 2:
                return '/'.join(f1_parts[-2:]) == '/'.join(f2_parts[-2:])
            return True

        return False

    def create_annotation_dataset(
        self,
        instance_ids: List[str],
        trace_results: Dict
    ) -> List[AnnotatedAssociation]:
        """
        Create a full annotation dataset for the given instances.
        This simulates what a human annotator would produce.
        """
        annotations = []

        for instance_id in instance_ids:
            ci_failure = self.log_details_dict.get(instance_id)
            if not ci_failure:
                continue

            # Get all possible problem-hunk pairs across all methods
            all_pairs = set()

            for method in ['endpoint', 'development', 'bidirectional']:
                # Would need to extract from trace_results
                pass

            # Annotate each unique pair
            # (Implementation would iterate through unique pairs)

        return annotations


def create_annotation_report(annotations: List[AnnotatedAssociation], output_path: Path):
    """Create a human-readable annotation report"""

    report_lines = [
        "="*80,
        "HUMAN ATTRIBUTION ANNOTATION REPORT",
        "="*80,
        "",
        f"Total Annotations: {len(annotations)}",
        ""
    ]

    # Group by association label
    by_label = {}
    for ann in annotations:
        label = ann.association_label
        by_label.setdefault(label, []).append(ann)

    report_lines.append("Distribution by Association Type:")
    report_lines.append("-"*80)
    for label in ['direct', 'shared', 'unrelated', 'uncertain']:
        count = len(by_label.get(label, []))
        pct = 100 * count / len(annotations) if annotations else 0
        report_lines.append(f"  {label:12s}: {count:4d} ({pct:5.1f}%)")

    report_lines.append("")
    report_lines.append("Distribution by Confidence:")
    report_lines.append("-"*80)

    by_confidence = {}
    for ann in annotations:
        conf = ann.confidence
        by_confidence.setdefault(conf, []).append(ann)

    for conf in ['high', 'medium', 'low']:
        count = len(by_confidence.get(conf, []))
        pct = 100 * count / len(annotations) if annotations else 0
        report_lines.append(f"  {conf:12s}: {count:4d} ({pct:5.1f}%)")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("DETAILED ANNOTATIONS")
    report_lines.append("="*80)

    # Show sample annotations
    for i, ann in enumerate(annotations[:20]):  # First 20 as examples
        report_lines.extend([
            "",
            f"[{i+1}] Instance {ann.instance_id} - Problem {ann.problem_id}",
            "-"*80,
            f"Problem: {ann.problem_description[:100]}...",
            f"File: {ann.file_path}:{ann.hunk_start}-{ann.hunk_end}",
            f"Evidence: {ann.failure_evidence[:100]}...",
            f"",
            f"ANNOTATION: {ann.association_label} (confidence: {ann.confidence})",
            f"Rationale: {ann.annotation_rationale}",
            ""
        ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Annotation report saved to: {output_path}")


if __name__ == "__main__":
    base_dir = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based"
    simulator = HumanAnnotationSimulator(base_dir)

    # Example usage would go here
    print("Human annotation simulator initialized.")
    print("Use this module to simulate detailed human review of problem-hunk associations.")
