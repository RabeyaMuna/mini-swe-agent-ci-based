"""
Memory Plugin - Agent-Agnostic Memory Retrieval
Handles decomposition caching and orchestrates memory retrieval.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .stair_retrieval import STAIRRetrieval
from .decomposition_cache import get_global_cache
from utilities.llm_invoker import STRICT_JSON_RULES, invoke_llm_with_retry


class DecompositionGenerationError(RuntimeError):
    """Raised when valid CI analysis cannot be decomposed into valid problems."""


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
        Retrieve CI failure repair solutions.

        Flow:
        1. Check decomposition cache (by sha_fail)
        2. If not cached: generate decomposition and save to cache
        3. If baseline: return problems (without query)
        4. If memory: pass to stair_retrieval for memory work

        Args:
            ci_failure: Complete CI failure analysis dict
            verification: Workflow verification dict
            issue_metadata: Metadata (workflow_path, workflow_name, repo, sha_fail)

        Returns:
            Dict with problems (and memory data if not baseline)
        """
        issue_metadata = issue_metadata or {}
        sha_fail = issue_metadata.get("sha_fail", "")

        # 1. Check cache for decomposed problems
        cache = get_global_cache()
        if sha_fail and cache.has(sha_fail):
            decomposed_problems = cache.get_problems(sha_fail)
            print(f"[Memory] Using cached decomposition for {sha_fail[:12]}")
        else:
            # 2. Generate decomposition and save to cache
            print(f"[Memory] Generating decomposition for {sha_fail[:12] if sha_fail else 'unknown'}")
            decomposed_problems = self._decompose_and_save_to_cache(
                ci_failure=ci_failure,
                verification=verification,
                workflow_path=issue_metadata.get("workflow_path", ""),
                workflow_name=issue_metadata.get("workflow_name", ""),
                repo=issue_metadata.get("repo", ""),
                sha_fail=sha_fail
            )

        # 3. Baseline mode: return problems without query
        if self.ablation.lower() == "baseline":
            return {"problems": decomposed_problems}

        # 4. Memory mode: pass to stair_retrieval for memory work
        # Pass metadata explicitly so retrieval doesn't have to extract from problems
        query_metadata = {
            "repo": issue_metadata.get("repo", ""),
            "workflow_path": issue_metadata.get("workflow_path", ""),
            "workflow_name": issue_metadata.get("workflow_name", ""),
        }
        return self.retrieval.retrieve(
            problems=decomposed_problems,
            top_k=self.top_k,
            query=query_metadata
        )

    def _decompose_and_save_to_cache(
        self,
        ci_failure: Dict[str, Any],
        verification: Optional[Dict[str, Any]],
        workflow_path: str,
        workflow_name: str,
        repo: str,
        sha_fail: str
    ) -> List[Dict[str, Any]]:
        """
        Decompose CI failure into problems and save to cache.

        Args:
            ci_failure: Complete CI failure analysis
            verification: Workflow verification
            workflow_path, workflow_name, repo: Metadata
            sha_fail: Failing commit SHA (for caching)

        Returns:
            List of decomposed problems with L1/L2/L3 queries
        """
        if not self.llm:
            raise ValueError("LLM client required for CI decomposition")

        # Validate we have CI failure data
        evidence_fields = ("error_context", "failure_signals", "relevant_files", "error_types")
        if not isinstance(ci_failure, dict) or not any(
            ci_failure.get(field) for field in evidence_fields
        ):
            raise DecompositionGenerationError(
                "CI log analysis is empty or malformed: expected at least one of "
                + ", ".join(evidence_fields)
            )

        # Extract workflow name if not provided
        if not workflow_name and workflow_path:
            workflow_name = Path(workflow_path).name

        # Build data for LLM
        compact_data = {
            "repo": repo,
            "workflow_name": workflow_name,
            "workflow_path": workflow_path,
            "ci_failure": ci_failure,
            "verification": verification or {}
        }

        prompt = f"""Decompose CI failure into structured problems with level-specific queries.

**Complete CI Failure Analysis:**
```json
{json.dumps(compact_data, indent=2)}
```

**Task:**
Analyze the CI failure and decompose it into individual problems.
For EACH problem, create structured query objects for L1/L2/L3 retrieval.

- If CI has ONE main failure → Return 1 problem
- If CI has MULTIPLE different failures → Return N problems (one per distinct issue)

**REQUIRED FIELDS:**
1. **problem**: Clear specific description (REQUIRED, cannot be empty/N/A)
2. **root_cause**: Explain WHY the failure happens (REQUIRED if failure_signals is empty)
3. **failure_signals**: Actual error messages/patterns from CI logs (REQUIRED if root_cause is empty, MUST be non-empty array)
   - Extract ACTUAL error messages from the CI log
   - Include stack traces, exception messages, test failures
   - DO NOT leave empty - use CI error_types evidence if needed
4. **files**: List of files to fix
5. **failure_type**: Category of failure

**IMPORTANT:** Every problem MUST have either a non-empty root_cause OR non-empty failure_signals array.
If CI log has error_types with evidence, use that evidence as failure_signals.

**Return JSON:**
```json
{{
  "problems": [
    {{
      "problem": "Clear specific description",
      "root_cause": "Why this failure happens",
      "files": ["file1.py", "file2.py"],
      "failure_type": "formatting",
      "failure_signals": ["error message 1", "error message 2"],
      "verification_cmd": "./test.sh",
      "query": {{
        "l1": {{
          "problem": "SAME as top-level problem",
          "root_cause": "SAME as top-level root_cause",
          "files": ["SAME as top-level files"],
          "failure_types": ["Category", "Sub-category"],
          "repo": "{repo}",
          "workflow_path": "{workflow_path}",
          "workflow_name": "{workflow_name}",
          "failure_signals": ["signal1", "signal2"]
        }},
        "l2": {{
          "problem": "SAME as l1",
          "root_cause": "SAME as l1",
          "files": ["SAME as l1"],
          "failure_types": ["SAME as l1"],
          "repo": "{repo}",
          "failure_signals": ["SAME as l1"]
        }},
        "l3": {{
          "problem": "Generic abstract version",
          "root_cause": "Generic version",
          "failure_types": ["SAME as l1"],
          "failure_signals": ["abstract patterns"]
        }}
      }}
    }}
  ]
}}
```

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)

        # Handle both formats
        if isinstance(response, list):
            raw_problems = response
        elif isinstance(response, dict):
            raw_problems = response.get("problems")
        else:
            raise DecompositionGenerationError(
                f"LLM returned malformed output: expected JSON object or array, got {type(response).__name__}"
            )

        if not isinstance(raw_problems, list) or not raw_problems:
            raise DecompositionGenerationError(
                "LLM returned no problems; the response must contain a non-empty 'problems' array"
            )

        problems = []
        for index, problem in enumerate(raw_problems, 1):
            # Validate problem has required structure
            if not isinstance(problem, dict):
                raise DecompositionGenerationError(
                    f"LLM returned malformed problem {index}: each problem must be an object"
                )

            # Validate problem description (required)
            problem_desc = str(problem.get("problem", "")).strip()
            if not problem_desc or len(problem_desc) <= 1 or problem_desc in ["", "N/A", "Unknown", "|"]:
                raise DecompositionGenerationError(
                    f"LLM returned malformed problem {index}: problem description is empty or invalid ('{problem_desc}')"
                )

            # Validate root_cause OR failure_signals (at least one required)
            root_cause = str(problem.get("root_cause", "")).strip()
            failure_signals = problem.get("failure_signals", [])
            has_root_cause = root_cause and root_cause not in ["", "N/A", "Unknown"]
            has_failure_signals = isinstance(failure_signals, list) and len(failure_signals) > 0

            if not (has_root_cause or has_failure_signals):
                # Provide detailed error message showing what the LLM actually returned
                print(f"[Memory] ❌ Problem {index} validation failed:")
                print(f"[Memory]    problem: {problem.get('problem', 'N/A')[:100]}...")
                print(f"[Memory]    root_cause: '{root_cause}'")
                print(f"[Memory]    failure_signals: {failure_signals}")
                raise DecompositionGenerationError(
                    f"LLM returned malformed problem {index}: must have either root_cause (non-empty) or failure_signals (non-empty array). "
                    f"Got root_cause='{root_cause}' and failure_signals={failure_signals}. "
                    f"The prompt explicitly requires these fields - this indicates the LLM is not following the schema."
                )

            problems.append(problem)

        # Save to cache
        if sha_fail and problems:
            cache = get_global_cache()
            model_name = getattr(self.llm, "model_name", "unknown")
            # Build minimal query for cache
            query = {"repo": repo, "workflow_name": workflow_name, "workflow_path": workflow_path}
            cache.set(sha_fail, query, problems, model=model_name)
            print(f"[Memory] Saved decomposition to cache for {sha_fail[:12]}")

        return problems

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
        workflow_path = (
            ci_failure.get("workflow_path")
            or verification.get("workflow_path")
            or issue_metadata.get("workflow_path")
            or ""
        )
        workflow_name = (
            ci_failure.get("workflow_name")
            or issue_metadata.get("workflow_name")
            or (Path(workflow_path).stem if workflow_path else "")
        )

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
            "workflow_name": workflow_name,
            "workflow_path": workflow_path,

            # ============================================================
            # Complete CI failure info (pass EVERYTHING for L1/L2/L3)
            # ============================================================
            "error_context": ci_failure.get("error_context", []),
            "failure_signals": ci_failure.get("failure_signals", []),
            "relevant_files": ci_failure.get("relevant_files", []),
            "error_types": ci_failure.get("error_types", []),
            # The CI log analyzer writes ``failed_job`` while some older
            # callers use ``failed_jobs``.  Accept both so a cached analyzer
            # result does not lose the failing command during decomposition.
            "failed_jobs": (
                ci_failure.get("failed_jobs")
                or ci_failure.get("failed_job")
                or []
            ),

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
                files = problem['files']
                lines.append("** Files to Modify:**")
                for f in files:
                    lines.append(f"  - `{f}`")
                lines.append("")

            # Failure signals
            if problem.get('failure_signals'):
                signals = problem['failure_signals']
                lines.append("** Error Signals:**")
                for sig in signals:
                    lines.append(f"  - {sig}")
                lines.append("")

            # Repair instructions
            repair = problem.get('repair_strategy') or {}
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
