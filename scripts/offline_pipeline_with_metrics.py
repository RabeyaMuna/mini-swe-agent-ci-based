#!/usr/bin/env python3
"""
offline_pipeline_with_metrics.py - Complete Offline Extraction Pipeline with Token/Cost Tracking
===============================================================================================

Runs the REAL offline extraction pipeline with comprehensive metrics tracking:
1. Log details extraction + CI workflow verification (via decompose_issue)
2. Problem decomposition (decompose_issue)
3. L1 memory building (generate_l1_from_decomposed_problems)
4. L2 memory building (build_l2_memory)
5. L3 memory building (build_l3_memory)

Tracks tokens, costs, and timing for each phase using RunMetricsRecorder.

Usage:
    # Process a single instance from dataset
    python scripts/offline_pipeline_with_metrics.py \
        --dataset data/cibench_test.json \
        --instance-id 102 \
        --model deepseek-v4-flash \
        --output-dir results/metrics/offline_extraction

    # Or use HuggingFace dataset
    python scripts/offline_pipeline_with_metrics.py \
        --use-huggingface \
        --instance-id 102 \
        --model deepseek-v4-flash \
        --output-dir results/metrics/offline_extraction

Output:
    - run_metrics.json - Complete token/cost breakdown
    - cost_time_report.json - Summary report
    - cost_time_report.csv - CSV export
    - detailed_usage_{instance_id}.json - Step-by-step breakdown
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backward_decomposition"))

# Import REAL modules from your codebase
from build_memory import (
    build_l2_memory,
    build_l3_memory,
    generate_l1_from_decomposed_problems,
)
from backward_decomposition.decompose_ci_failure import decompose_issue
from utilities.llm_model import LitellmModel
from utilities.model_registry import configure_model_environment
from utilities.run_metrics import RunMetricsRecorder
from datasets import load_dataset


def load_issue_from_dataset(
    instance_id: str,
    dataset_path: Optional[Path] = None,
    use_huggingface: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Load a specific issue from the dataset.

    Args:
        instance_id: The instance ID to load
        dataset_path: Path to local JSON/JSONL file
        use_huggingface: Whether to load from HuggingFace

    Returns:
        The issue dict, or None if not found
    """
    if use_huggingface:
        print(f"Loading from HuggingFace: RabeyaMuna/CI-Bench...")
        dataset = load_dataset("RabeyaMuna/CI-Bench", split="test")
        issues = list(dataset)
    elif dataset_path and dataset_path.exists():
        print(f"Loading from: {dataset_path}")
        with open(dataset_path) as f:
            if dataset_path.suffix == '.jsonl':
                issues = [json.loads(line) for line in f]
            else:
                data = json.load(f)
                if isinstance(data, list):
                    issues = data
                elif isinstance(data, dict) and "instances" in data:
                    issues = data["instances"]
                else:
                    raise ValueError(f"Unknown dataset format in {dataset_path}")
    else:
        raise ValueError("Must provide either --dataset or --use-huggingface")

    # Find the issue by ID
    for issue in issues:
        issue_id = str(
            issue.get("id")
            or issue.get("instance_id")
            or issue.get("issue_id")
            or ""
        )
        if issue_id == instance_id:
            return issue

    return None


class OfflineExtractionPipeline:
    """
    Orchestrates the complete offline extraction pipeline with metrics tracking.
    Uses the REAL backward_decomposition and build_memory modules.
    """

    def __init__(
        self,
        issue: Dict[str, Any],
        instance_id: str,
        model: str,
        output_dir: Path,
        skip_l3: bool = False,
    ):
        self.issue = issue
        self.instance_id = instance_id
        self.model_name = model
        self.output_dir = output_dir
        self.skip_l3 = skip_l3

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.output_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)

        # Initialize metrics recorder
        self.metrics = RunMetricsRecorder(
            output_dir=str(self.output_dir),
            instance_id=instance_id,
            agent="offline_extraction",
            model=model,
            direction="backward",
            ablation="L1+L2+L3" if not skip_l3 else "L1+L2",
        )

        # Initialize LLM with metrics tracking
        self.llm = LitellmModel(
            model_name=self.model_name,
            metrics_recorder=self.metrics,
        )

    def run_phase(self, phase_name: str, phase_func):
        """
        Run a pipeline phase with metrics tracking.

        Args:
            phase_name: Name of the phase (e.g., "decomposition", "build_l1")
            phase_func: Function to execute for this phase

        Returns:
            Result from phase_func
        """
        print(f"\n{'=' * 80}")
        print(f"Phase: {phase_name}")
        print(f"{'=' * 80}")

        # Set the current phase on the LLM for metrics tracking
        self.llm._current_phase = phase_name

        start_time = time.time()
        try:
            result = phase_func()
            elapsed = time.time() - start_time
            print(f"OK {phase_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"FAIL {phase_name} failed after {elapsed:.2f}s: {e}")
            raise
        finally:
            # Reset phase
            self.llm._current_phase = "context_llm"

    def phase_endpoint_view(self) -> Dict[str, Any]:
        """
        Phase 1: Endpoint View - CI log reconstruction and workflow verification.

        NOTE: In practice, these are usually CACHED from prior extraction,
        so this phase has minimal LLM cost.
        """
        print("Building endpoint view (CI logs + workflow)...")

        # Import the internal function
        from backward_decomposition.decompose_ci_failure import build_benchmark_ci_context

        # Set phase for metrics tracking
        self.llm._current_phase = "endpoint_view_reconstruction"

        benchmark_context = build_benchmark_ci_context(
            issue=self.issue,
            llm=self.llm,
            output_dir=str(self.output_dir),
        )

        print(f"  Validation sequence: {len(benchmark_context.get('validation_sequence', []))} steps")
        return benchmark_context

    def phase_development_and_reconciliation(self, benchmark_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2 & 3: Development View + Cross-View Reconciliation.

        - Development View: Classify code changes by validation
        - Cross-View Reconciliation: Match CI failures to code changes
        """
        print("Analyzing development view + cross-view reconciliation...")

        # Import the internal function
        from backward_decomposition.decompose_ci_failure import (
            analyze_diff_chunks,
            _cluster_and_merge_problems,
            _reorder_by_repair_trajectory,
            _issue_id,
        )

        # This function internally does BOTH:
        # - classify_chunk_with_fallback (development view)
        # - analyze_validation_groups_with_reasoning (reconciliation)
        # We'll track them together for now since they're in one function
        self.llm._current_phase = "development_view_and_reconciliation"

        diff_context = analyze_diff_chunks(
            issue=self.issue,
            benchmark_context=benchmark_context,
            llm=self.llm,
            output_dir=str(self.output_dir),
        )

        atomic_problems = diff_context.get("atomic_problems", [])

        if not atomic_problems:
            raise ValueError("No atomic problems identified")

        print(f"  Identified {len(atomic_problems)} atomic problems")

        # Step 4: Similarity-based grouping + LLM merge
        self.llm._current_phase = "problem_deduplication"
        optimized_problems = atomic_problems
        if len(atomic_problems) > 1:
            print("  Deduplicating similar problems...")
            optimized_problems = _cluster_and_merge_problems(
                atomic_problems,
                validation_cmd="backward_deduplication",
                llm=self.llm,
                similarity_threshold=0.85,
            )
            merged = len(atomic_problems) - len(optimized_problems)
            if merged > 0:
                print(f"    Merged {merged} redundant problems")

        # Step 5: Reorder
        print("  Reordering by repair trajectory...")
        final_problems = _reorder_by_repair_trajectory(optimized_problems)

        # Build final result
        issue_id = _issue_id(self.issue) or self.instance_id
        context = benchmark_context.get("context", {})
        log_analysis = benchmark_context.get("log_analysis", {})

        result = {
            "original_issue_id": issue_id,
            "sha_fail": self.issue.get("sha_fail"),
            "repo": self.issue.get("repo_name", self.issue.get("repo")),
            "original_error_type": self.issue.get("error_type"),
            "problems": final_problems,
            "total_problems": len(final_problems),
            "raw_atomic_problems_count": len(atomic_problems),
            "total_changed_files": len(self.issue.get("changed_files", [])),
            "benchmark_ci_context": {
                "workflow_path": benchmark_context.get("workflow_path"),
                "workflow_name": benchmark_context.get("workflow_name"),
                "validation_sequence": benchmark_context.get("validation_sequence", []),
                "overall_failure_reasons": context.get("overall_failure_reasons", []),
                "overall_error_types": context.get("overall_error_types", []),
                "error_types": log_analysis.get("error_types", []),
                "relevant_files": log_analysis.get("relevant_files", []),
                "failed_jobs": log_analysis.get("failed_job", []),
            },
            "diff_analysis_context": {
                "mode": diff_context.get("mode"),
                "total_files": diff_context.get("total_files", 0),
                "total_changes": diff_context.get("total_changes", 0),
                "chunk_count": diff_context.get("chunk_count", 0),
                "validation_groups_count": diff_context.get("validation_groups", {}).get("total_groups", 0),
                "validation_groups": diff_context.get("validation_groups", {}).get("validation_groups", {}),
            },
        }

        # Save decomposed result
        decomposed_path = self.output_dir / f"decomposed_{self.instance_id}.json"
        with open(decomposed_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  Final: {len(final_problems)} problems")
        print(f"  Saved to: {decomposed_path}")

        return result

    def phase_build_l1(self, decomposed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Build L1 memory (concrete problem-level failures).

        Uses the REAL generate_l1_from_decomposed_problems function.
        """
        print("Building L1 memory using generate_l1_from_decomposed_problems...")

        # Use the REAL L1 builder with correct parameters
        repo = decomposed_result.get("repo", "unknown")
        repo_parts = repo.split("/") if "/" in repo else [repo, ""]

        l1_memory = generate_l1_from_decomposed_problems(
            issue_id=decomposed_result.get("original_issue_id", self.instance_id),
            repo=repo,
            repo_owner=repo_parts[0] if len(repo_parts) > 0 else "unknown",
            workflow_path=decomposed_result.get("benchmark_ci_context", {}).get("workflow_path", ""),
            decomposed_problems=decomposed_result.get("problems", []),
            dependencies=decomposed_result.get("dependencies", {}),
            ground_truth_files=decomposed_result.get("changed_files", []),
        )

        # Save L1 memory
        l1_path = self.memory_dir / "failure_memory.json"
        existing_l1 = []
        if l1_path.exists():
            with open(l1_path) as f:
                existing_l1 = json.load(f)
                if not isinstance(existing_l1, list):
                    existing_l1 = [existing_l1]

        existing_l1.append(l1_memory)
        with open(l1_path, "w") as f:
            json.dump(existing_l1, f, indent=2)

        print(f"L1 memory saved to: {l1_path}")
        return l1_memory

    def phase_build_l2(self, l1_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: Build L2 memory (repair strategies).

        Uses the REAL build_l2_memory function.
        """
        print("Building L2 memory using build_l2_memory...")

        # Use the REAL L2 builder
        l2_memory = build_l2_memory(
            l1_memory=l1_memory,
            llm=self.llm,
        )

        # Save L2 memory
        l2_path = self.memory_dir / "repo_memory.json"
        existing_l2 = []
        if l2_path.exists():
            with open(l2_path) as f:
                existing_l2 = json.load(f)
                if not isinstance(existing_l2, list):
                    existing_l2 = [existing_l2]

        existing_l2.append(l2_memory)
        with open(l2_path, "w") as f:
            json.dump(existing_l2, f, indent=2)

        print(f"L2 memory saved to: {l2_path}")
        return l2_memory

    def phase_build_l3(
        self, l1_memory: Dict[str, Any], l2_memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 4: Build L3 memory (universal patterns).

        Uses the REAL build_l3_memory function.
        """
        print("Building L3 memory using build_l3_memory...")

        # Use the REAL L3 builder
        l3_memory = build_l3_memory(
            l1_memory=l1_memory,
            l2_memory=l2_memory,
            llm=self.llm,
        )

        # Save L3 patterns
        l3_path = self.memory_dir / "cross_memory.json"
        existing_l3 = []
        if l3_path.exists():
            with open(l3_path) as f:
                existing_l3 = json.load(f)
                if not isinstance(existing_l3, list):
                    existing_l3 = [existing_l3]

        # Extract patterns
        patterns = l3_memory.get("patterns", [])
        existing_l3.extend(patterns)

        with open(l3_path, "w") as f:
            json.dump(existing_l3, f, indent=2)

        print(f"L3 memory saved to: {l3_path}")
        return l3_memory

    def run(self):
        """
        Execute the complete offline extraction pipeline.
        """
        print(f"\n{'=' * 80}")
        print(f"OFFLINE EXTRACTION PIPELINE WITH METRICS")
        print(f"{'=' * 80}")
        print(f"Instance ID: {self.instance_id}")
        print(f"Repository: {self.issue.get('repo', 'unknown')}")
        print(f"Model: {self.model_name}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 80}\n")

        try:
            # Phase 1: Endpoint View (CI log + workflow reconstruction)
            # NOTE: Usually cached, minimal cost
            benchmark_context = self.run_phase(
                "endpoint_view_reconstruction",
                self.phase_endpoint_view,
            )

            # Phase 2 & 3: Development View + Cross-View Reconciliation
            # Development: Classify code changes by validation
            # Reconciliation: Match CI failures to code changes
            decomposed_result = self.run_phase(
                "development_view_and_reconciliation",
                lambda: self.phase_development_and_reconciliation(benchmark_context),
            )

            # Phase 4: Build L1 (concrete problems)
            # This is mostly data transformation, minimal API calls
            l1_memory = self.run_phase(
                "build_l1",
                lambda: self.phase_build_l1(decomposed_result),
            )

            # Phase 5: Build L2 (repair strategies)
            # This uses LLM to generate strategies
            l2_memory = self.run_phase(
                "build_l2",
                lambda: self.phase_build_l2(l1_memory),
            )

            # Phase 6: Build L3 (universal patterns) - if not skipped
            # This uses LLM to extract patterns
            if not self.skip_l3:
                l3_memory = self.run_phase(
                    "build_l3",
                    lambda: self.phase_build_l3(l1_memory, l2_memory),
                )

            # Finalize metrics
            self.metrics.finish(status="completed")

            print(f"\n{'=' * 80}")
            print("PIPELINE COMPLETE")
            print(f"{'=' * 80}")
            print(f"Metrics saved to: {self.output_dir}/run_metrics.json")
            print(f"Report saved to: {self.output_dir}/cost_time_report.json")
            print(f"CSV report: {self.output_dir}/cost_time_report.csv")
            print(f"{'=' * 80}\n")

        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"PIPELINE FAILED: {e}")
            print(f"{'=' * 80}\n")
            self.metrics.finish(status="failed", error=str(e))
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Run offline extraction pipeline with comprehensive metrics tracking"
    )
    parser.add_argument(
        "--instance-id",
        required=True,
        help="Instance ID to process",
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset JSON/JSONL file",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Load from HuggingFace RabeyaMuna/CI-Bench",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model to use (e.g., deepseek-v4-flash, glm5.2)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metrics/offline_extraction",
        help="Output directory for results and metrics",
    )
    parser.add_argument(
        "--skip-l3",
        action="store_true",
        help="Skip L3 generation (only build L1 and L2)",
    )

    args = parser.parse_args()

    # Load the issue
    dataset_path = Path(args.dataset) if args.dataset else None
    issue = load_issue_from_dataset(
        instance_id=args.instance_id,
        dataset_path=dataset_path,
        use_huggingface=args.use_huggingface,
    )

    if not issue:
        print(f"ERROR: Instance {args.instance_id} not found in dataset")
        return 1

    # Configure model environment
    model_name = configure_model_environment(args.model) or args.model

    # Create and run pipeline
    pipeline = OfflineExtractionPipeline(
        issue=issue,
        instance_id=args.instance_id,
        model=model_name,
        output_dir=Path(args.output_dir),
        skip_l3=args.skip_l3,
    )

    pipeline.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
