"""
Problem-Hunk Attribution Evaluation for ReCAP
Implements the algorithm from the paper's evaluation methodology
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import random


class AssociationType(Enum):
    """Types of problem-hunk associations"""
    DIRECT = "direct"        # Hunk directly fixes the problem
    SHARED = "shared"        # Hunk contributes to fixing (multi-problem scenario)
    UNRELATED = "unrelated"  # Hunk does not address this problem
    UNCERTAIN = "uncertain"  # Cannot confidently determine


@dataclass
class CIProblem:
    """A distinct CI problem identified in the failure"""
    problem_id: str
    description: str
    failure_type: str  # e.g., 'test_failure', 'build_error', 'lint_error'
    test_name: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_key(self) -> str:
        """Unique identifier for this problem"""
        return self.problem_id


@dataclass
class CodeHunk:
    """A canonical code change hunk"""
    hunk_id: str
    file_path: str
    start_line: int
    end_line: int
    diff_content: str
    commit_sha: Optional[str] = None

    def to_key(self) -> str:
        """Unique identifier for this hunk"""
        return self.hunk_id


@dataclass
class ProblemHunkPair:
    """A problem-hunk association pair"""
    problem: CIProblem
    hunk: CodeHunk

    def to_key(self) -> Tuple[str, str]:
        """Unique identifier for this pair"""
        return (self.problem.to_key(), self.hunk.to_key())

    def __hash__(self):
        return hash(self.to_key())

    def __eq__(self, other):
        return self.to_key() == other.to_key()


@dataclass
class LLMAnnotation:
    """LLM-proposed association"""
    pair: ProblemHunkPair
    proposed_type: AssociationType
    explanation: str
    confidence: float


@dataclass
class HumanAnnotation:
    """Human-verified association"""
    pair: ProblemHunkPair
    association_type: AssociationType
    notes: str
    annotator_id: str


@dataclass
class RepairEpisode:
    """A complete fail-to-pass repair episode"""
    episode_id: str
    repository: str

    # Input: Historical data
    ci_logs: str
    workflow_config: str
    commit_history: List[Dict]

    # Extracted components
    problems: List[CIProblem] = field(default_factory=list)
    hunks: List[CodeHunk] = field(default_factory=list)

    # ReCAP predictions
    recap_associations: Set[ProblemHunkPair] = field(default_factory=set)

    # Ground truth annotations
    llm_annotations: List[LLMAnnotation] = field(default_factory=list)
    human_annotations: List[HumanAnnotation] = field(default_factory=list)

    # Metadata
    num_problems: int = 0
    num_hunks: int = 0
    patch_size: int = 0
    failure_types: Set[str] = field(default_factory=set)


@dataclass
class EvaluationResult:
    """Results for attribution evaluation"""
    total_episodes: int

    # Micro-averaged (all pairs)
    micro_precision: float
    micro_recall: float
    micro_f1: float

    # Counts
    total_tp: int
    total_fp: int
    total_fn: int

    # Stratified results
    single_problem_metrics: Dict[str, float]
    multi_problem_metrics: Dict[str, float]

    # Per-episode breakdown
    per_episode_results: List[Dict]

    # Uncertainty analysis
    uncertain_pairs_count: int
    uncertain_pairs_examples: List[Dict]


class ProblemHunkEvaluator:
    """
    Evaluates problem-hunk attribution following the paper's algorithm
    """

    def __init__(self):
        self.episodes: List[RepairEpisode] = []

    def stratified_sample(
        self,
        all_histories: List[RepairEpisode],
        n: int = 50
    ) -> List[RepairEpisode]:
        """
        Step 1: Stratified sampling of repair episodes

        Stratification criteria:
        - Single vs multi-problem histories
        - Different failure types
        - Different repositories
        - Different patch sizes
        """
        # Categorize histories
        single_problem = [h for h in all_histories if h.num_problems == 1]
        multi_problem = [h for h in all_histories if h.num_problems > 1]

        # Size categories
        small_patch = [h for h in all_histories if h.patch_size < 50]
        medium_patch = [h for h in all_histories if 50 <= h.patch_size < 200]
        large_patch = [h for h in all_histories if h.patch_size >= 200]

        # Target distribution (example)
        target = {
            'single_problem': int(n * 0.4),  # 40% single-problem
            'multi_problem': int(n * 0.6),   # 60% multi-problem
            'small_patch': int(n * 0.3),
            'medium_patch': int(n * 0.5),
            'large_patch': int(n * 0.2)
        }

        # Sample with stratification
        # (Simplified - in practice, use sklearn.model_selection.StratifiedShuffleSplit)
        sample = []

        # Ensure diversity
        sample.extend(random.sample(single_problem, min(target['single_problem'], len(single_problem))))
        sample.extend(random.sample(multi_problem, min(target['multi_problem'], len(multi_problem))))

        # De-duplicate and take top N
        sample = list(set(sample))[:n]

        return sample

    def identify_ci_problems(self, episode: RepairEpisode) -> List[CIProblem]:
        """
        Step 3a: Identify distinct CI problems from failure logs

        Parse CI logs and workflow evidence to extract individual problems
        """
        problems = []

        # This is placeholder - actual implementation would parse real CI logs
        # Example patterns:
        # - Test failures: Each failed test is one problem
        # - Build errors: Each compilation error is one problem
        # - Lint errors: Group related lint issues

        # Example extraction logic:
        if "FAILED" in episode.ci_logs:
            # Parse test failures
            # problems.append(CIProblem(...))
            pass

        return problems

    def extract_canonical_hunks(self, episode: RepairEpisode) -> List[CodeHunk]:
        """
        Step 3b: Extract canonical changed hunks from repair history

        - Split hunks containing unrelated logical edits
        - Reconcile repeated edits to the same code
        """
        hunks = []

        for commit in episode.commit_history:
            # Parse diff to extract hunks
            # Split if hunk contains multiple logical changes
            # Merge if same location edited multiple times
            pass

        return hunks

    def generate_llm_annotations(
        self,
        problems: List[CIProblem],
        hunks: List[CodeHunk],
        episode: RepairEpisode
    ) -> List[LLMAnnotation]:
        """
        Step 3d: Independent LLM proposes candidate associations

        IMPORTANT: LLM has NO ACCESS to ReCAP predictions
        Uses only: CI logs, workflow configs, commit history, code changes
        """
        annotations = []

        # For each possible problem-hunk pair
        for problem in problems:
            for hunk in hunks:
                pair = ProblemHunkPair(problem, hunk)

                # Create prompt for independent LLM
                prompt = self._create_llm_prompt(problem, hunk, episode)

                # Get LLM response (placeholder - use actual LLM API)
                # response = llm_api.complete(prompt)

                # Parse response
                annotation = LLMAnnotation(
                    pair=pair,
                    proposed_type=AssociationType.DIRECT,  # placeholder
                    explanation="LLM reasoning here",
                    confidence=0.85
                )

                annotations.append(annotation)

        return annotations

    def _create_llm_prompt(
        self,
        problem: CIProblem,
        hunk: CodeHunk,
        episode: RepairEpisode
    ) -> str:
        """Create prompt for independent LLM annotation"""

        prompt = f"""You are analyzing a CI failure repair to determine if a code change addresses a specific problem.

**CI Problem:**
- Type: {problem.failure_type}
- Description: {problem.description}
- Test: {problem.test_name or 'N/A'}
- Error: {problem.error_message or 'N/A'}

**Code Change Hunk:**
- File: {hunk.file_path}
- Lines: {hunk.start_line}-{hunk.end_line}
- Diff:
```diff
{hunk.diff_content}
```

**Context:**
- Repository: {episode.repository}
- CI Logs: [excerpt]
- Workflow: [excerpt]

**Task:**
Determine the association between this problem and this code hunk:

1. **DIRECT**: This hunk directly fixes this specific problem
2. **SHARED**: This hunk contributes to fixing this problem (possibly among others)
3. **UNRELATED**: This hunk does not address this problem
4. **UNCERTAIN**: Cannot confidently determine from available evidence

**Important:**
- You do NOT have access to any external system's predictions
- Base your judgment ONLY on the evidence provided
- Be conservative: if unsure, mark as UNCERTAIN

Output JSON:
{{
  "association_type": "direct|shared|unrelated|uncertain",
  "confidence": 0.0-1.0,
  "explanation": "Your detailed reasoning"
}}
"""
        return prompt

    def human_verification(
        self,
        all_pairs: Set[ProblemHunkPair],
        llm_annotations: List[LLMAnnotation],
        episode: RepairEpisode,
        annotator_id: str
    ) -> List[HumanAnnotation]:
        """
        Step 3e: Human reviews ALL pairs and finalizes ground truth

        - Review every possible pair (including LLM omissions)
        - Correct incorrect LLM associations
        - Identify missing associations
        - Label each pair
        """
        human_annotations = []

        # Create lookup of LLM annotations
        llm_lookup = {ann.pair.to_key(): ann for ann in llm_annotations}

        # Review ALL pairs (not just LLM-proposed)
        for pair in all_pairs:
            llm_ann = llm_lookup.get(pair.to_key())

            # Present to human annotator
            # In practice, this would be a UI showing:
            # - Problem details
            # - Hunk diff
            # - LLM's proposed association (if any)
            # - Context (CI logs, etc.)

            # Human makes final decision
            # (Placeholder - in practice, from UI input)
            final_type = AssociationType.DIRECT  # placeholder
            notes = "Human verification notes"

            annotation = HumanAnnotation(
                pair=pair,
                association_type=final_type,
                notes=notes,
                annotator_id=annotator_id
            )

            human_annotations.append(annotation)

        return human_annotations

    def normalize_and_align(
        self,
        recap_predictions: Set[ProblemHunkPair],
        canonical_problems: List[CIProblem],
        canonical_hunks: List[CodeHunk]
    ) -> Set[ProblemHunkPair]:
        """
        Step 3f: Map ReCAP associations to canonical problem-hunk pairs

        IMPORTANT: Do NOT use ground truth to modify predictions
        Only normalize IDs and align to canonical representations
        """
        aligned = set()

        # Create lookup tables
        problem_map = {p.problem_id: p for p in canonical_problems}
        hunk_map = {h.hunk_id: h for h in canonical_hunks}

        for pair in recap_predictions:
            # Map to canonical representations
            # Handle cases where ReCAP uses different IDs or splits/merges hunks
            canonical_pair = ProblemHunkPair(
                problem=problem_map.get(pair.problem.problem_id, pair.problem),
                hunk=hunk_map.get(pair.hunk.hunk_id, pair.hunk)
            )
            aligned.add(canonical_pair)

        return aligned

    def evaluate_episode(
        self,
        episode: RepairEpisode,
        human_annotations: List[HumanAnnotation],
        recap_predictions: Set[ProblemHunkPair]
    ) -> Tuple[int, int, int, int]:
        """
        Step 3h-i: Calculate TP, FP, FN for one episode

        Returns: (TP, FP, FN, uncertain_count)
        """
        # Ground truth positive pairs (direct or shared, excluding uncertain)
        G_pos = set()
        uncertain_pairs = set()

        for ann in human_annotations:
            if ann.association_type == AssociationType.UNCERTAIN:
                uncertain_pairs.add(ann.pair)
            elif ann.association_type in [AssociationType.DIRECT, AssociationType.SHARED]:
                G_pos.add(ann.pair)
            # UNRELATED pairs are implicitly negative (not in G_pos)

        # ReCAP predicted positive associations
        R_pos = recap_predictions

        # Calculate metrics (excluding uncertain)
        R_pos_certain = R_pos - uncertain_pairs

        TP = len(R_pos_certain & G_pos)
        FP = len(R_pos_certain - G_pos)
        FN = len(G_pos - R_pos_certain)

        return TP, FP, FN, len(uncertain_pairs)

    def evaluate_batch(
        self,
        episodes: List[RepairEpisode]
    ) -> EvaluationResult:
        """
        Steps 4-6: Calculate micro-averaged metrics across all episodes
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_uncertain = 0

        per_episode_results = []

        single_problem_tp, single_problem_fp, single_problem_fn = 0, 0, 0
        multi_problem_tp, multi_problem_fp, multi_problem_fn = 0, 0, 0

        for episode in episodes:
            tp, fp, fn, uncertain = self.evaluate_episode(
                episode,
                episode.human_annotations,
                episode.recap_associations
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_uncertain += uncertain

            # Stratify by problem count
            if episode.num_problems == 1:
                single_problem_tp += tp
                single_problem_fp += fp
                single_problem_fn += fn
            else:
                multi_problem_tp += tp
                multi_problem_fp += fp
                multi_problem_fn += fn

            # Per-episode metrics
            episode_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            episode_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            episode_f1 = 2 * episode_p * episode_r / (episode_p + episode_r) if (episode_p + episode_r) > 0 else 0.0

            per_episode_results.append({
                'episode_id': episode.episode_id,
                'precision': episode_p,
                'recall': episode_r,
                'f1': episode_f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'uncertain': uncertain
            })

        # Step 4-6: Micro-averaged metrics
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Stratified metrics
        single_p = single_problem_tp / (single_problem_tp + single_problem_fp) if (single_problem_tp + single_problem_fp) > 0 else 0.0
        single_r = single_problem_tp / (single_problem_tp + single_problem_fn) if (single_problem_tp + single_problem_fn) > 0 else 0.0
        single_f1 = 2 * single_p * single_r / (single_p + single_r) if (single_p + single_r) > 0 else 0.0

        multi_p = multi_problem_tp / (multi_problem_tp + multi_problem_fp) if (multi_problem_tp + multi_problem_fp) > 0 else 0.0
        multi_r = multi_problem_tp / (multi_problem_tp + multi_problem_fn) if (multi_problem_tp + multi_problem_fn) > 0 else 0.0
        multi_f1 = 2 * multi_p * multi_r / (multi_p + multi_r) if (multi_p + multi_r) > 0 else 0.0

        return EvaluationResult(
            total_episodes=len(episodes),
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            single_problem_metrics={
                'precision': single_p,
                'recall': single_r,
                'f1': single_f1
            },
            multi_problem_metrics={
                'precision': multi_p,
                'recall': multi_r,
                'f1': multi_f1
            },
            per_episode_results=per_episode_results,
            uncertain_pairs_count=total_uncertain,
            uncertain_pairs_examples=[]  # Could collect examples
        )


def save_results(result: EvaluationResult, output_path: str):
    """Save evaluation results to JSON"""
    data = asdict(result)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    print("""
    Problem-Hunk Attribution Evaluation
    ====================================

    This script implements the evaluation algorithm from the paper.

    Steps:
    1. Load your ReCAP system's predictions
    2. Run stratified sampling
    3. For each episode:
       a. Extract CI problems
       b. Extract canonical hunks
       c. Generate LLM annotations (independent)
       d. Human verification (UI or CLI)
    4. Calculate metrics

    See evaluation_workflow.md for complete instructions.
    """)
