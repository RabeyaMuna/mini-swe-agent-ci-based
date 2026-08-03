"""
Memory Plugin - Agent-Agnostic Memory Retrieval
Handles query building and memory retrieval for any CI repair agent.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from .stair_retrieval import STAIRRetrieval


class MemoryPlugin:
    """
    Agent-agnostic memory plugin for CI repair.

    Responsibilities:
    - Build query from raw CI failure and verification data
    - Retrieve similar past fixes from L1/L2/L3 memory
    - Organize results for agent consumption

    The plugin handles ALL query building logic so agents only need to:
    1. Load CI failure and verification
    2. Pass raw data to this plugin
    3. Format the returned results for their specific agent
    """

    def __init__(
        self,
        memory_root: Path,
        result_dir: str,
        ablation: str = "L1+L2+L3",
        top_k: int = 5,
        llm: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize memory plugin.

        Args:
            memory_root: Path to L1/L2/L3 memory files (e.g., data/back_trs)
            result_dir: Directory to save retrieval results
            ablation: Which memory levels to use (baseline, L1, L1+L2, L1+L2+L3)
            top_k: Number of results to retrieve per level
            llm: Optional LLM client for advanced filtering
            enabled: Enable/disable memory (False for baseline mode)
        """
        self.memory_root = str(memory_root)
        self.result_dir = result_dir
        self.ablation = ablation
        self.top_k = top_k
        self.llm = llm
        self.enabled = enabled

        # Initialize STAIR retrieval system
        self.retrieval = STAIRRetrieval(
            memory_dir=self.memory_root,
            llm_client=llm,
            baseline_mode=(not enabled),
            memory_levels=ablation
        )

    def retrieve(
        self,
        ci_failure: Dict[str, Any],
        verification: Optional[Dict[str, Any]] = None,
        issue_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve similar past fixes from memory.

        Args:
            ci_failure: Complete CI failure analysis dict containing:
                - error_context: List[str]
                - failure_signals: List[str]
                - relevant_files: List[dict]
                - error_types: List[dict]
                - failed_jobs: List[dict]
                - workflow_name: str
                - workflow_path: str
            verification: Workflow verification dict containing:
                - validation_sequence: List[dict]
            issue_metadata: Optional metadata for logging (NOT used in similarity):
                - task_id: str
                - sha_fail: str
                - repo: str

        Returns:
            Dict containing:
                - problems: List of organized problems from memory
                - metadata: Retrieval statistics
                - l1_matches: Raw L1 matches
                - l2_matches: Raw L2 matches
                - l3_matches: Raw L3 matches
                - level_scores: Similarity scores per level
        """
        # Baseline check: return empty if baseline mode
        if not self.enabled or self.ablation.lower() == "baseline":
            return {
                'problems': [],
                'metadata': {
                    'mode': 'baseline',
                    'enabled_levels': [],
                    'retrieved': {'l1': 0, 'l2': 0, 'l3': 0},
                    'common_detected': 0,
                    'filtered': 0,
                    'merged': 0,
                    'final': 0
                },
                'l1_matches': [],
                'l2_matches': [],
                'l3_matches': [],
                'common_problems': [],
                'dependencies': []
            }

        # Build query from raw CI data
        query = self._build_query(ci_failure, verification, issue_metadata)

        # Do retrieval
        retrieval_result = self.retrieval.retrieve(query, top_k=self.top_k)

        # Add raw query to result for debugging
        retrieval_result['query'] = query

        return retrieval_result

    def _build_query(
        self,
        ci_failure: Dict[str, Any],
        verification: Optional[Dict[str, Any]],
        issue_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build query for L1/L2/L3 retrieval from raw CI failure data.

        This is the central query building logic - ALL agents use this.

        Query Structure:
        - Metadata (for logging only, NOT for similarity search)
        - Repository info (for L1/L2 filtering)
        - Complete CI failure info (for all levels)
        - Validation info (for repair strategies)

        L1 uses: repo + workflow + ALL CI failure info
        L2 uses: repo + ALL CI failure info (cross-workflow)
        L3 uses: ONLY semantic features (error types, categories, tools)
        """
        verification = verification or {}
        issue_metadata = issue_metadata or {}

        # Extract failed commands from validation
        failed_cmd = self._extract_failed_commands(verification)

        # Normalize repo name (handle both "owner/repo" and "repo" formats)
        repo_raw = issue_metadata.get("repo", "")
        repo_normalized = repo_raw.split("/")[-1] if "/" in repo_raw else repo_raw

        return {
            # ============================================================
            # Metadata (for logging/tracking, NOT for similarity search)
            # ============================================================
            "task_id": issue_metadata.get("task_id", ""),
            "sha_fail": issue_metadata.get("sha_fail", ""),

            # ============================================================
            # Repository info (for L1/L2 filtering)
            # ============================================================
            "repo": repo_normalized,  # Use normalized name for matching
            "workflow_name": ci_failure.get("workflow_name", ""),
            "workflow_path": ci_failure.get("workflow_path", ""),

            # ============================================================
            # Complete CI failure info (pass EVERYTHING for L1/L2/L3)
            # ============================================================
            "error_context": ci_failure.get("error_context", []),
            "failure_signals": ci_failure.get("failure_signals", []),
            "relevant_files": ci_failure.get("relevant_files", []),
            "error_types": ci_failure.get("error_types", []),
            "failed_jobs": ci_failure.get("failed_jobs", []),

            # ============================================================
            # Validation info (for repair strategies)
            # ============================================================
            "validation_sequence": verification.get("validation_sequence", []),
            "failed_cmd": failed_cmd,
        }

    def _extract_failed_commands(self, verification: Dict[str, Any]) -> List[str]:
        """Extract failed commands from validation sequence."""
        validation_sequence = verification.get("validation_sequence", [])
        return [
            step.get("validation_cmd")
            for step in validation_sequence
            if isinstance(step, dict) and step.get("validation_cmd")
        ]

    def format_for_prompt(self, retrieval: Dict[str, Any]) -> str:
        """
        Format retrieval results as markdown for agent prompt.

        This is a generic formatter - agents can override for their format.

        Args:
            retrieval: Result from retrieve() method

        Returns:
            Markdown-formatted memory context
        """
        problems = retrieval.get('problems', [])
        metadata = retrieval.get('metadata', {})

        if not problems:
            return "## Memory Context\n\nNo similar past fixes found in memory."

        lines = [
            "## CI Fix Instructions - Memory-Guided Repair",
            "",
            f"You must fix **all {len(problems)} problems** below to resolve the CI failure.",
            "Each problem includes complete repair instructions from similar past fixes.",
            "",
            "**Guidelines:**",
            "- Fix problems in the order listed (dependencies first)",
            "- Follow the action steps precisely",
            "- Run validation command after each fix",
            "- If validation fails due to setup issues, document and continue",
            "- If new problems arise during fixes, address them as well",
            "",
            "---",
            "",
        ]

        for i, problem in enumerate(problems, 1):  # Show ALL problems (removed limit)
            lines.append(f"## Problem {i} of {len(problems)}")
            lines.append("")

            # Problem description
            if problem.get('problem'):
                lines.append(f"**Problem**: {problem['problem']}")
                lines.append("")

            # Root cause
            if problem.get('root_cause'):
                lines.append(f"**Root Cause**: {problem['root_cause']}")
                lines.append("")

            # Files to modify
            if problem.get('files'):
                files = problem['files'][:10]
                lines.append("** Files to Modify:**")
                for f in files:
                    lines.append(f"  - `{f}`")
                lines.append("")

            # Failure signals
            if problem.get('failure_signals'):
                signals = problem['failure_signals'][:5]
                lines.append("** Error Signals:**")
                for sig in signals:
                    lines.append(f"  - {sig}")
                lines.append("")

            # Repair instructions
            repair = problem.get('repair_strategy', {})
            if repair.get('summary'):
                lines.append("** Repair Approach:**")
                lines.append(f"{repair['summary']}")
                lines.append("")

            if repair.get('actions'):
                lines.append("** Step-by-Step Actions:**")
                for idx, action in enumerate(repair['actions'][:15], 1):
                    lines.append(f"{idx}. {action}")
                lines.append("")

            if repair.get('pitfalls'):
                lines.append("**WARNING Pitfalls to Avoid:**")
                for pitfall in repair['pitfalls'][:5]:
                    lines.append(f"  - {pitfall}")
                lines.append("")

            # Validation
            if repair.get('validation_cmd'):
                lines.append("**OK Validation Command:**")
                lines.append(f"```bash")
                lines.append(f"{repair['validation_cmd']}")
                lines.append(f"```")
                lines.append("*Note: If validation fails due to setup issues (missing dependencies, environment problems), document the issue and continue with remaining fixes.*")
                lines.append("")

            # Source levels
            source = problem.get('source', {})
            if source:
                sources = []
                if source.get('l1'):
                    sources.append(f"L1({len(source['l1'])})")
                if source.get('l2'):
                    sources.append(f"L2({len(source['l2'])})")
                if source.get('l3'):
                    sources.append(f"L3({len(source['l3'])})")
                if sources:
                    lines.append(f"**Source**: {' + '.join(sources)}")
                    lines.append("")

            lines.append("---")
            lines.append("")

        # Add statistics
        lines.append("### Retrieval Statistics")
        lines.append("")
        lines.append(f"- **Enabled Levels**: {', '.join(metadata.get('enabled_levels', []))}")
        retrieved = metadata.get('retrieved', {})
        lines.append(f"- **Retrieved**: L1={retrieved.get('l1', 0)}, L2={retrieved.get('l2', 0)}, L3={retrieved.get('l3', 0)}")
        lines.append(f"- **Common Problems Detected**: {metadata.get('common_detected', 0)}")
        lines.append(f"- **Final Problems**: {len(problems)}")

        return '\n'.join(lines)
