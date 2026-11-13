#!/usr/bin/env python3
"""
offline_pipeline_with_three_views.py - Complete Pipeline with THREE-VIEW Separation
====================================================================================

Tracks costs separately for:
1. Endpoint View (Backward from CI failure)
2. Development View (Forward from commits/code changes)
3. Cross-View Reconciliation (Bidirectional CI ↔ Code matching)
4. L1 Build (Data transformation)
5. L2 Build (Contextual strategies)
6. L3 Build (Transferable patterns)

This replaces offline_pipeline_with_metrics.py with proper three-view tracking.
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

from build_memory import (
    build_l2_memory,
    build_l3_memory,
    generate_l1_from_decomposed_problems,
)
from backward_decomposition.decompose_ci_failure import (
    build_benchmark_ci_context,
    parse_diff_to_structured,
    build_dependency_graph_from_structured_diff,
    chunk_structured_diff,
    _get_model_aware_limits,
    classify_chunk_with_fallback,
    merge_chunks_by_validation,
    _compact_context_for_diff_analysis,
    analyze_validation_groups_with_reasoning,
    _cluster_and_merge_problems,
    _reorder_by_repair_trajectory,
    _issue_id,
)
from utilities.llm_model import LitellmModel
from utilities.model_registry import configure_model_environment
from utilities.run_metrics import RunMetricsRecorder
from datasets import load_dataset


def load_issue_from_dataset(
    instance_id: str,
    dataset_path: Optional[Path] = None,
    use_huggingface: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load a specific issue from the dataset."""
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


class ThreeViewPipeline:
    """
    Orchestrates the complete offline extraction pipeline with THREE-VIEW tracking:

    1. Endpoint View (Backward from CI)
    2. Development View (Forward from commits)
    3. Cross-View Reconciliation (Bidirectional CI ↔ Code)
    4. L1/L2/L3 Memory Building
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
            agent="offline_extraction_three_views",
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
        """Run a pipeline phase with metrics tracking."""
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
        VIEW 1: Endpoint View (Backward from CI failure)

        Analyzes CI logs and workflow to understand what failed.
        Usually CACHED, so minimal cost.
        """
        print("VIEW 1: Endpoint View (Backward from CI failure)...")

        benchmark_context = build_benchmark_ci_context(
            issue=self.issue,
            llm=self.llm,
            output_dir=str(self.output_dir),
        )

        print(f"  Validation sequence: {len(benchmark_context.get('validation_sequence', []))} steps")
        return benchmark_context

    def phase_development_view(self, benchmark_context: Dict[str, Any]) -> tuple:
        """
        VIEW 2: Development View (Forward from commits/code changes)

        Analyzes code changes to understand what was modified.
        """
        print("VIEW 2: Development View (Forward from commits)...")

        # Get diff
        diff = str(self.issue.get("diff") or "")
        if not diff.strip():
            raise ValueError(f"Issue {self.instance_id} has no ground-truth diff")

        # Step 0: Parse diff (deterministic, no LLM)
        print("  Parsing diff into structured format...")
        structured_diff = parse_diff_to_structured(diff)
        total_files = structured_diff["total_files"]
        total_changes = structured_diff["total_changes"]
        print(f"    {total_files} files, {total_changes} changes")

        # Step 0.5: Build dependency graph (deterministic, no LLM)
        print("  Building file dependency graph...")
        dependency_graph = build_dependency_graph_from_structured_diff(
            structured_diff, repo_path=benchmark_context.get("repo_path")
        )
        print(f"    {len(dependency_graph.get('clusters', []))} dependency clusters")

        # Chunk the diff
        model_limits = _get_model_aware_limits(self.model_name)
        chunks = chunk_structured_diff(
            structured_diff,
            max_files_per_chunk=model_limits["max_files_per_chunk"],
            dependency_graph=dependency_graph,
            model_name=self.model_name,
        )

        # CLASSIFY CODE CHANGES BY VALIDATION (Development View)
        print(f"  Classifying {total_files} files in {len(chunks)} chunks...")

        visible_failure_context = _compact_context_for_diff_analysis(
            self.issue, benchmark_context
        )
        validation_sequence = benchmark_context.get("validation_sequence") or []
        chunk_findings = []

        for index, chunk in enumerate(chunks, start=1):
            validations = classify_chunk_with_fallback(
                chunk=chunk,
                chunk_index=index,
                total_chunks=len(chunks),
                visible_failure_context=visible_failure_context,
                validation_sequence=validation_sequence,
                llm=self.llm,
            )
            if validations:
                chunk_findings.append(
                    {"chunk_index": index, "validations_in_this_chunk": validations}
                )

        # Merge by validation (deterministic)
        print("  Merging chunks by validation...")
        validation_groups = merge_chunks_by_validation(
            chunk_findings, validation_sequence, chunks
        )
        print(f"    {validation_groups['total_groups']} validation groups")

        return (validation_groups, validation_sequence, total_files, total_changes, chunks)

    def phase_reconciliation(
        self,
        benchmark_context: Dict[str, Any],
        validation_groups: Dict,
        validation_sequence: list,
    ) -> Dict[str, Any]:
        """
        VIEW 3: Cross-View Reconciliation (Bidirectional CI ↔ Code matching)

        Matches CI failures to code changes to identify root causes.
        """
        print("VIEW 3: Cross-View Reconciliation (Bidirectional CI ↔ Code)...")

        ci_context = _compact_context_for_diff_analysis(self.issue, benchmark_context)

        print("  Deep reasoning with CI-Code correlation...")
        reasoning_result = analyze_validation_groups_with_reasoning(
            validation_groups,
            validation_sequence,
            ci_context,
            self.llm,
            output_dir=str(self.output_dir),
        )

        atomic_problems = reasoning_result.get("atomic_problems", [])
        print(f"  Identified {len(atomic_problems)} atomic problems")

        return atomic_problems

    def phase_deduplication(self, atomic_problems: list) -> list:
        """Deduplicate similar problems."""
        print("Deduplicating similar problems...")

        if len(atomic_problems) <= 1:
            return atomic_problems

        optimized_problems = _cluster_and_merge_problems(
            atomic_problems,
            validation_cmd="backward_deduplication",
            llm=self.llm,
            similarity_threshold=0.85,
        )

        merged = len(atomic_problems) - len(optimized_problems)
        if merged > 0:
            print(f"  Merged {merged} redundant problems")

        return optimized_problems

    def finalize_decomposition(
        self,
        problems: list,
        benchmark_context: Dict,
        total_files: int,
        total_changes: int,
        validation_groups: Dict,
    ) -> Dict[str, Any]:
        """Finalize and save decomposition result."""

        # Reorder
        print("Reordering by repair trajectory...")
        final_problems = _reorder_by_repair_trajectory(problems)

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
            "total_changed_files": total_files,
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
                "total_files": total_files,
                "total_changes": total_changes,
                "validation_groups_count": validation_groups.get("total_groups", 0),
            },
        }

        # Save
        decomposed_path = self.output_dir / f"decomposed_{self.instance_id}.json"
        with open(decomposed_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Final: {len(final_problems)} problems")
        print(f"Saved to: {decomposed_path}")

        return result

    def phase_build_l1(self, decomposed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build L1 memory (data transformation only)."""
        print("Building L1 memory...")

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

        # Save
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
        """Build L2 memory (repair strategies)."""
        print("Building L2 memory...")

        l2_memory = build_l2_memory(
            l1_memory=l1_memory,
            llm=self.llm,
        )

        # Save
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
        """Build L3 memory (universal patterns)."""
        print("Building L3 memory...")

        l3_memory = build_l3_memory(
            l1_memory=l1_memory,
            l2_memory=l2_memory,
            llm=self.llm,
        )

        # Save
        l3_path = self.memory_dir / "cross_memory.json"
        existing_l3 = []
        if l3_path.exists():
            with open(l3_path) as f:
                existing_l3 = json.load(f)
                if not isinstance(existing_l3, list):
                    existing_l3 = [existing_l3]

        patterns = l3_memory.get("patterns", [])
        existing_l3.extend(patterns)

        with open(l3_path, "w") as f:
            json.dump(existing_l3, f, indent=2)

        print(f"L3 memory saved to: {l3_path}")
        return l3_memory

    def run(self):
        """Execute the complete pipeline with THREE-VIEW tracking."""
        print(f"\n{'=' * 80}")
        print(f"THREE-VIEW OFFLINE EXTRACTION PIPELINE")
        print(f"{'=' * 80}")
        print(f"Instance ID: {self.instance_id}")
        print(f"Repository: {self.issue.get('repo', 'unknown')}")
        print(f"Model: {self.model_name}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 80}\n")

        try:
            # VIEW 1: Endpoint (Backward from CI)
            benchmark_context = self.run_phase(
                "endpoint_view_backward",
                self.phase_endpoint_view,
            )

            # VIEW 2: Development (Forward from commits)
            dev_result = self.run_phase(
                "development_view_commit_based",
                lambda: self.phase_development_view(benchmark_context),
            )
            validation_groups, validation_sequence, total_files, total_changes, chunks = dev_result

            # VIEW 3: Reconciliation (Bidirectional CI ↔ Code)
            atomic_problems = self.run_phase(
                "cross_view_reconciliation_bidirectional",
                lambda: self.phase_reconciliation(benchmark_context, validation_groups, validation_sequence),
            )

            # Problem Deduplication
            optimized_problems = self.run_phase(
                "problem_deduplication",
                lambda: self.phase_deduplication(atomic_problems),
            )

            # Finalize
            decomposed_result = self.finalize_decomposition(
                optimized_problems,
                benchmark_context,
                total_files,
                total_changes,
                validation_groups,
            )

            # L1 Build (no LLM)
            l1_memory = self.run_phase(
                "build_l1",
                lambda: self.phase_build_l1(decomposed_result),
            )

            # L2 Build
            l2_memory = self.run_phase(
                "build_l2",
                lambda: self.phase_build_l2(l1_memory),
            )

            # L3 Build (if not skipped)
            if not self.skip_l3:
                self.run_phase(
                    "build_l3",
                    lambda: self.phase_build_l3(l1_memory, l2_memory),
                )

            # Finalize metrics
            self.metrics.finish(status="completed")

            print(f"\n{'=' * 80}")
            print("PIPELINE COMPLETE")
            print(f"{'=' * 80}")
            print(f"Metrics: {self.output_dir}/run_metrics.json")
            print(f"Report: {self.output_dir}/cost_time_report.json")
            print(f"{'=' * 80}\n")

        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"PIPELINE FAILED: {e}")
            print(f"{'=' * 80}\n")
            self.metrics.finish(status="failed", error=str(e))
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Run offline extraction pipeline with THREE-VIEW tracking"
    )
    parser.add_argument("--instance-id", required=True, help="Instance ID to process")
    parser.add_argument("--dataset", help="Path to dataset JSON/JSONL file")
    parser.add_argument("--use-huggingface", action="store_true", help="Load from HuggingFace")
    parser.add_argument("--model", required=True, help="LLM model to use")
    parser.add_argument("--output-dir", default="results/metrics/three_view_extraction", help="Output directory")
    parser.add_argument("--skip-l3", action="store_true", help="Skip L3 generation")

    args = parser.parse_args()

    # Load the issue
    dataset_path = Path(args.dataset) if args.dataset else None
    issue = load_issue_from_dataset(
        instance_id=args.instance_id,
        dataset_path=dataset_path,
        use_huggingface=args.use_huggingface,
    )

    if not issue:
        print(f"ERROR: Instance {args.instance_id} not found")
        return 1

    # Configure model
    model_name = configure_model_environment(args.model) or args.model

    # Run pipeline
    pipeline = ThreeViewPipeline(
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
