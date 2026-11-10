"""
Complete End-to-End Annotation Pipeline
Orchestrates LLM annotation + Human verification workflow
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from problem_hunk_evaluation import (
    RepairEpisode, CIProblem, CodeHunk, ProblemHunkPair,
    LLMAnnotation, HumanAnnotation, AssociationType,
    ProblemHunkEvaluator, save_results
)


class AnnotationPipeline:
    """Orchestrates the complete annotation workflow"""

    def __init__(
        self,
        work_dir: str = "./annotation_work",
        llm_api_key: Optional[str] = None
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)

        self.evaluator = ProblemHunkEvaluator()
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")

        # Directories
        self.episodes_dir = self.work_dir / "episodes"
        self.llm_annotations_dir = self.work_dir / "llm_annotations"
        self.human_annotations_dir = self.work_dir / "human_annotations"
        self.results_dir = self.work_dir / "results"

        for d in [self.episodes_dir, self.llm_annotations_dir,
                  self.human_annotations_dir, self.results_dir]:
            d.mkdir(exist_ok=True)

    def load_episodes_from_recap_output(
        self,
        recap_output_file: str
    ) -> List[RepairEpisode]:
        """
        Load repair episodes from ReCAP system output

        Expected format:
        {
          "episodes": [
            {
              "episode_id": "...",
              "repository": "...",
              "ci_logs": "...",
              "workflow_config": "...",
              "commit_history": [...],
              "recap_predictions": [
                {"problem_id": "...", "hunk_id": "..."},
                ...
              ]
            }
          ]
        }
        """
        with open(recap_output_file, 'r') as f:
            data = json.load(f)

        episodes = []
        for ep_data in data['episodes']:
            episode = RepairEpisode(
                episode_id=ep_data['episode_id'],
                repository=ep_data['repository'],
                ci_logs=ep_data['ci_logs'],
                workflow_config=ep_data.get('workflow_config', ''),
                commit_history=ep_data.get('commit_history', [])
            )

            # Load ReCAP predictions
            # (You'll need to parse these into actual ProblemHunkPair objects)
            # episode.recap_associations = ...

            episodes.append(episode)

        return episodes

    def step1_stratified_sampling(
        self,
        all_episodes: List[RepairEpisode],
        n: int = 50
    ) -> List[RepairEpisode]:
        """Step 1: Select stratified sample"""
        print(f"Step 1: Stratified sampling ({n} episodes from {len(all_episodes)})")

        sample = self.evaluator.stratified_sample(all_episodes, n)

        # Save sample list
        sample_file = self.work_dir / "sample_episodes.json"
        with open(sample_file, 'w') as f:
            json.dump([ep.episode_id for ep in sample], f, indent=2)

        print(f"  ✓ Selected {len(sample)} episodes")
        print(f"  ✓ Saved to {sample_file}")

        return sample

    def step2_extract_problems_and_hunks(
        self,
        episode: RepairEpisode
    ) -> RepairEpisode:
        """Step 2: Extract CI problems and canonical hunks"""
        print(f"\nProcessing episode: {episode.episode_id}")

        # Extract problems
        print("  Extracting CI problems...")
        episode.problems = self.evaluator.identify_ci_problems(episode)
        episode.num_problems = len(episode.problems)
        print(f"    Found {episode.num_problems} distinct problems")

        # Extract hunks
        print("  Extracting canonical hunks...")
        episode.hunks = self.evaluator.extract_canonical_hunks(episode)
        episode.num_hunks = len(episode.hunks)
        print(f"    Found {episode.num_hunks} code hunks")

        # Save extracted data
        ep_file = self.episodes_dir / f"{episode.episode_id}.json"
        self._save_episode(episode, ep_file)

        return episode

    def step3_llm_annotation(
        self,
        episode: RepairEpisode,
        use_cache: bool = True
    ) -> List[LLMAnnotation]:
        """Step 3: Independent LLM generates candidate annotations"""
        cache_file = self.llm_annotations_dir / f"{episode.episode_id}.json"

        if use_cache and cache_file.exists():
            print(f"  Loading cached LLM annotations from {cache_file}")
            return self._load_llm_annotations(cache_file)

        print(f"  Generating LLM annotations...")
        print(f"    Total pairs to evaluate: {len(episode.problems)} × {len(episode.hunks)} = {len(episode.problems) * len(episode.hunks)}")

        annotations = self.evaluator.generate_llm_annotations(
            episode.problems,
            episode.hunks,
            episode
        )

        # Filter by confidence threshold (optional)
        high_confidence = [a for a in annotations if a.confidence >= 0.7]
        print(f"    Generated {len(annotations)} annotations ({len(high_confidence)} high confidence)")

        # Save LLM annotations
        self._save_llm_annotations(annotations, cache_file)

        return annotations

    def step4_human_verification(
        self,
        episode: RepairEpisode,
        llm_annotations: List[LLMAnnotation],
        annotator_id: str = "annotator_1",
        use_ui: bool = True
    ) -> List[HumanAnnotation]:
        """Step 4: Human verifies and corrects annotations"""
        cache_file = self.human_annotations_dir / f"{episode.episode_id}_{annotator_id}.json"

        if cache_file.exists():
            response = input(f"  Found existing human annotations. Use them? (y/n): ")
            if response.lower() == 'y':
                return self._load_human_annotations(cache_file)

        print(f"  Starting human verification...")

        # Generate all possible pairs
        all_pairs = {
            ProblemHunkPair(p, h)
            for p in episode.problems
            for h in episode.hunks
        }

        print(f"    Total pairs to review: {len(all_pairs)}")

        if use_ui:
            # Launch web UI
            print(f"    Launching annotation UI...")
            print(f"    → Open: http://localhost:5000")
            print(f"    → Episode: {episode.episode_id}")
            # self._launch_annotation_ui(episode, llm_annotations)
            print("    (UI not implemented in this example - using CLI mode)")
            use_ui = False

        if not use_ui:
            # CLI mode
            human_annotations = self._cli_annotation_interface(
                all_pairs,
                llm_annotations,
                episode,
                annotator_id
            )

        # Save human annotations
        self._save_human_annotations(human_annotations, cache_file)

        return human_annotations

    def _cli_annotation_interface(
        self,
        all_pairs: set,
        llm_annotations: List[LLMAnnotation],
        episode: RepairEpisode,
        annotator_id: str
    ) -> List[HumanAnnotation]:
        """Simple CLI interface for human annotation"""
        print("\n" + "="*80)
        print("HUMAN ANNOTATION MODE (CLI)")
        print("="*80)
        print("\nInstructions:")
        print("  For each problem-hunk pair, label as:")
        print("    d = DIRECT (hunk directly fixes this problem)")
        print("    s = SHARED (hunk contributes to fix)")
        print("    u = UNRELATED (hunk doesn't address this problem)")
        print("    ? = UNCERTAIN (can't determine)")
        print("    a = ACCEPT LLM suggestion")
        print("    q = QUIT (save progress)")
        print("="*80)

        llm_lookup = {ann.pair.to_key(): ann for ann in llm_annotations}
        human_annotations = []

        for i, pair in enumerate(all_pairs, 1):
            print(f"\n--- Pair {i}/{len(all_pairs)} ---")
            print(f"\nPROBLEM: {pair.problem.description}")
            print(f"  Type: {pair.problem.failure_type}")
            if pair.problem.test_name:
                print(f"  Test: {pair.problem.test_name}")

            print(f"\nHUNK: {pair.hunk.file_path}:{pair.hunk.start_line}-{pair.hunk.end_line}")
            print(f"  Diff:\n{pair.hunk.diff_content[:200]}...")

            # Show LLM suggestion
            llm_ann = llm_lookup.get(pair.to_key())
            if llm_ann:
                print(f"\nLLM SUGGESTION: {llm_ann.proposed_type.value} (conf: {llm_ann.confidence:.2f})")
                print(f"  Reasoning: {llm_ann.explanation}")
            else:
                print("\nLLM SUGGESTION: (none)")

            # Get human input
            while True:
                choice = input("\nYour label [d/s/u/?/a/q]: ").strip().lower()

                if choice == 'q':
                    print("Saving progress and exiting...")
                    return human_annotations

                if choice == 'a' and llm_ann:
                    association_type = llm_ann.proposed_type
                    notes = f"Accepted LLM suggestion: {llm_ann.explanation}"
                    break
                elif choice == 'd':
                    association_type = AssociationType.DIRECT
                    notes = input("  Optional notes: ")
                    break
                elif choice == 's':
                    association_type = AssociationType.SHARED
                    notes = input("  Optional notes: ")
                    break
                elif choice == 'u':
                    association_type = AssociationType.UNRELATED
                    notes = input("  Optional notes: ")
                    break
                elif choice == '?':
                    association_type = AssociationType.UNCERTAIN
                    notes = input("  Why uncertain? ")
                    break
                else:
                    print("  Invalid choice. Try again.")

            annotation = HumanAnnotation(
                pair=pair,
                association_type=association_type,
                notes=notes,
                annotator_id=annotator_id
            )
            human_annotations.append(annotation)

            print(f"  ✓ Labeled as: {association_type.value}")

        return human_annotations

    def step5_evaluate(
        self,
        episodes: List[RepairEpisode]
    ):
        """Step 5: Calculate metrics and generate report"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        result = self.evaluator.evaluate_batch(episodes)

        # Print results
        print(f"\nDataset: {result.total_episodes} episodes")
        print(f"\n--- Micro-Averaged Metrics (Problem-Hunk Level) ---")
        print(f"Precision: {result.micro_precision:.1%}")
        print(f"Recall:    {result.micro_recall:.1%}")
        print(f"F1 Score:  {result.micro_f1:.1%}")

        print(f"\n--- Counts ---")
        print(f"True Positives:  {result.total_tp}")
        print(f"False Positives: {result.total_fp}")
        print(f"False Negatives: {result.total_fn}")
        print(f"Uncertain Pairs: {result.uncertain_pairs_count} (excluded from primary metrics)")

        print(f"\n--- Stratified Results ---")
        print(f"Single-Problem Episodes:")
        print(f"  Precision: {result.single_problem_metrics['precision']:.1%}")
        print(f"  Recall:    {result.single_problem_metrics['recall']:.1%}")
        print(f"  F1 Score:  {result.single_problem_metrics['f1']:.1%}")

        print(f"\nMulti-Problem Episodes:")
        print(f"  Precision: {result.multi_problem_metrics['precision']:.1%}")
        print(f"  Recall:    {result.multi_problem_metrics['recall']:.1%}")
        print(f"  F1 Score:  {result.multi_problem_metrics['f1']:.1%}")

        # Save detailed results
        results_file = self.results_dir / "attribution_evaluation_results.json"
        save_results(result, str(results_file))
        print(f"\n✓ Detailed results saved to {results_file}")

        return result

    def run_complete_pipeline(
        self,
        recap_output_file: str,
        n_episodes: int = 50,
        annotator_id: str = "annotator_1"
    ):
        """Run the complete end-to-end pipeline"""
        print("="*80)
        print("RECAP ATTRIBUTION EVALUATION PIPELINE")
        print("="*80)

        # Load all episodes
        print("\nLoading ReCAP outputs...")
        all_episodes = self.load_episodes_from_recap_output(recap_output_file)
        print(f"  Loaded {len(all_episodes)} episodes")

        # Step 1: Stratified sampling
        sample = self.step1_stratified_sampling(all_episodes, n_episodes)

        # Process each episode
        for i, episode in enumerate(sample, 1):
            print(f"\n{'='*80}")
            print(f"Episode {i}/{len(sample)}: {episode.episode_id}")
            print(f"{'='*80}")

            # Step 2: Extract problems and hunks
            episode = self.step2_extract_problems_and_hunks(episode)

            # Step 3: LLM annotation
            llm_annotations = self.step3_llm_annotation(episode)
            episode.llm_annotations = llm_annotations

            # Step 4: Human verification
            human_annotations = self.step4_human_verification(
                episode,
                llm_annotations,
                annotator_id
            )
            episode.human_annotations = human_annotations

        # Step 5: Evaluate
        result = self.step5_evaluate(sample)

        return result

    # Helper methods for saving/loading
    def _save_episode(self, episode: RepairEpisode, path: Path):
        # Simplified - would need proper serialization
        pass

    def _save_llm_annotations(self, annotations: List[LLMAnnotation], path: Path):
        # Simplified
        pass

    def _load_llm_annotations(self, path: Path) -> List[LLMAnnotation]:
        # Simplified
        return []

    def _save_human_annotations(self, annotations: List[HumanAnnotation], path: Path):
        # Simplified
        pass

    def _load_human_annotations(self, path: Path) -> List[HumanAnnotation]:
        # Simplified
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ReCAP attribution evaluation pipeline"
    )
    parser.add_argument(
        "--recap-output",
        required=True,
        help="Path to ReCAP system output file"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate (default: 50)"
    )
    parser.add_argument(
        "--work-dir",
        default="./annotation_work",
        help="Working directory for annotations"
    )
    parser.add_argument(
        "--annotator-id",
        default="annotator_1",
        help="Annotator identifier"
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = AnnotationPipeline(work_dir=args.work_dir)
    result = pipeline.run_complete_pipeline(
        recap_output_file=args.recap_output,
        n_episodes=args.n_episodes,
        annotator_id=args.annotator_id
    )

    print("\n✓ Pipeline complete!")
    print(f"Final Results: P={result.micro_precision:.1%}, R={result.micro_recall:.1%}, F1={result.micro_f1:.1%}")
