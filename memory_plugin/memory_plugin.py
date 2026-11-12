"""
Memory Plugin - Agent-Agnostic Memory Retrieval
Handles decomposition caching and orchestrates memory retrieval.
"""

import json
import re
import shlex
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
        enabled: bool = True,
        metrics_recorder: Optional[Any] = None
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
            metrics_recorder: Optional RunMetricsRecorder for tracking API calls
        """
        self.memory_root = str(memory_root)
        self.result_dir = result_dir
        self.ablation = ablation
        self.top_k = top_k
        self.llm = llm
        self.enabled = enabled
        self.metrics_recorder = metrics_recorder

        # Read component ablation flags from environment variables
        import os
        enable_dependency_expansion = os.environ.get("RECAP_ENABLE_DEPENDENCY_EXPANSION", "1") == "1"
        enable_recurring_recall = os.environ.get("RECAP_ENABLE_RECURRING_RECALL", "1") == "1"
        enable_filtering = os.environ.get("RECAP_ENABLE_FILTERING", "1") == "1"
        enable_ordering = os.environ.get("RECAP_ENABLE_ORDERING", "1") == "1"
        enable_deduplication = os.environ.get("RECAP_ENABLE_DEDUPLICATION", "1") == "1"
        enable_memory_optimization = os.environ.get("RECAP_ENABLE_MEMORY_OPTIMIZATION", "1") == "1"
        enable_evidence_gate = os.environ.get("RECAP_ENABLE_EVIDENCE_GATE", "1") == "1"
        self.enable_source_validation = (
            os.environ.get("RECAP_ENABLE_SOURCE_VALIDATION", "1") == "1"
        )

        # Initialize STAIR retrieval system
        self.retrieval = STAIRRetrieval(
            memory_dir=self.memory_root,
            llm_client=llm,
            baseline_mode=(not enabled),
            memory_levels=ablation,
            metrics_recorder=metrics_recorder,
            enable_dependency_expansion=enable_dependency_expansion,
            enable_recurring_recall=enable_recurring_recall,
            enable_filtering=enable_filtering,
            enable_ordering=enable_ordering,
            enable_deduplication=enable_deduplication,
            enable_memory_optimization=enable_memory_optimization,
            enable_evidence_gate=enable_evidence_gate,
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

        # 4. Memory mode: pass the complete target evidence to retrieval.
        # Passing only repo/workflow makes applicability filtering blind to the
        # current log, failed tool, and affected files.
        query_metadata = self._build_query(
            ci_failure=ci_failure,
            verification=verification,
            issue_metadata=issue_metadata,
        )
        return self.retrieval.retrieve(
            problems=decomposed_problems,
            top_k=self.top_k,
            query=query_metadata
        )

    def validate_candidates_against_checkout(
        self,
        retrieval: Dict[str, Any],
        checkout_path: Path,
        command_runner: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Withhold memory hypotheses that are not supported by target source.

        This gate runs after ``sha_fail`` has been checked out and before the
        repair agent is constructed.  Problems observed in the target CI log
        remain actionable.  Recurring/dependency/common problems are actionable
        only when their claimed location exists and an independent read-only
        source review confirms that the described defect is present there.

        An unavailable model, absent location, malformed response, or ambiguous
        evidence is fail-closed: the candidate is retained in diagnostics but
        is not sent to the repair agent.
        """
        if not self.enable_source_validation or not retrieval:
            return retrieval

        root = Path(checkout_path).resolve()
        result = dict(retrieval)
        kept: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = list(
            retrieval.get("source_rejected_memory_problems") or []
        )

        # A dependency/hidden failure may not appear in the current log because
        # CI stopped at an earlier failure.  Therefore source validation also
        # reviews hypotheses withheld by the log-evidence gate.  They can be
        # promoted only by a positive source confirmation.
        candidates = list(retrieval.get("problems") or [])
        candidates.extend(retrieval.get("rejected_memory_problems") or [])
        unique_candidates = []
        seen_candidates = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            identity = (
                str(candidate.get("problem", "")).strip().lower(),
                tuple(self._candidate_files(candidate)),
                str(candidate.get("problem_type", "")).strip().lower(),
            )
            if identity in seen_candidates:
                continue
            seen_candidates.add(identity)
            unique_candidates.append(candidate)

        for index, original in enumerate(unique_candidates, 1):
            problem = dict(original)
            if self._is_observed_target_problem(problem):
                problem["source_validation"] = {
                    "decision": "observed",
                    "reason": "Problem is directly supported by the target CI failure evidence.",
                }
                kept.append(problem)
                continue

            claimed_files = self._candidate_files(problem)
            if not claimed_files:
                rejected.append(self._with_source_decision(
                    problem,
                    "unverifiable",
                    "Memory candidate names no target source location.",
                    [],
                ))
                continue

            resolved = []
            for claimed in claimed_files[:8]:
                matched = self._resolve_checkout_file(root, claimed)
                if matched and matched not in resolved:
                    resolved.append(matched)

            if not resolved:
                rejected.append(self._with_source_decision(
                    problem,
                    "rejected",
                    "None of the claimed files exists in the failure-state checkout.",
                    [],
                ))
                continue

            command_check = self._run_lightweight_verification(
                problem, root, resolved, command_runner
            )
            if command_check["status"] == "problem_absent":
                annotated = self._with_source_decision(
                    problem,
                    "rejected",
                    "The lightweight verification command passes at sha_fail; the remembered problem is absent.",
                    [command_check.get("command", "")],
                )
                annotated["source_validation"]["lightweight_check"] = command_check
                rejected.append(annotated)
                continue
            if command_check["status"] == "failure_confirmed":
                annotated = self._with_source_decision(
                    problem,
                    "confirmed",
                    "The lightweight verification command reproduces the candidate's failure signal at sha_fail.",
                    command_check.get("matched_signals") or [],
                )
                annotated["source_validation"]["lightweight_check"] = command_check
                kept.append(annotated)
                continue

            evidence = self._read_source_evidence(root, resolved, problem)
            if not evidence or not self.llm:
                reason = (
                    "The claimed files could not be read as source evidence."
                    if not evidence
                    else "No independent source-validation model is configured."
                )
                rejected.append(self._with_source_decision(
                    problem, "unverifiable", reason,
                    [str(path.relative_to(root)) for path in resolved],
                ))
                continue

            decision = self._review_candidate_source(
                problem, evidence, index, command_check
            )
            annotated = self._with_source_decision(
                problem,
                decision["decision"],
                decision["reason"],
                decision.get("evidence") or [],
            )
            annotated["source_validation"]["lightweight_check"] = command_check
            if decision["decision"] == "confirmed":
                kept.append(annotated)
            else:
                rejected.append(annotated)

        result["problems"] = kept
        result["source_rejected_memory_problems"] = rejected
        result["source_validation"] = {
            "enabled": True,
            "checkout": str(root),
            "input_count": len(unique_candidates),
            "actionable_count": len(kept),
            "withheld_count": len(rejected),
            "policy": "Only target-CI problems and source-confirmed memory candidates are actionable.",
        }
        return result

    @staticmethod
    def _verification_command(problem: Dict[str, Any]) -> str:
        strategy = problem.get("repair_strategy")
        strategy = strategy if isinstance(strategy, dict) else {}
        return str(
            problem.get("verification_cmd")
            or problem.get("validation_cmd")
            or strategy.get("validation_cmd")
            or strategy.get("verification_cmd")
            or ""
        ).strip()

    def _run_lightweight_verification(
        self,
        problem: Dict[str, Any],
        root: Path,
        resolved: List[Path],
        command_runner: Optional[Any],
    ) -> Dict[str, Any]:
        """Run only bounded, non-installing validation commands.

        A non-zero result is not automatically evidence: missing tools,
        dependency/setup failures, and unrelated test failures are inconclusive.
        """
        command = self._verification_command(problem)
        safe_command, reason = self._safe_lightweight_command(command, root, resolved)
        if not command_runner or not safe_command:
            return {
                "status": "skipped",
                "command": command,
                "reason": reason or "No command runner is available.",
            }

        try:
            result = command_runner(safe_command, timeout=45)
        except Exception as exc:
            return {
                "status": "inconclusive",
                "command": safe_command,
                "reason": f"Command execution failed: {type(exc).__name__}",
            }
        if not isinstance(result, dict):
            return {
                "status": "inconclusive",
                "command": safe_command,
                "reason": "Command runner returned malformed output.",
            }

        returncode = int(result.get("returncode", -1))
        output = str(result.get("output") or "")[:12000]
        summary = {
            "command": safe_command,
            "returncode": returncode,
            "output_excerpt": output[:3000],
        }
        if returncode == 0:
            return {**summary, "status": "problem_absent", "reason": "Command passed."}

        infrastructure_markers = (
            "command not found", "no module named", "module not found",
            "modulenotfounderror", "not installed", "no such file or directory",
            "could not find a version", "failed to build", "connection error",
            "network is unreachable", "timed out", "collected 0 items",
            "unrecognized arguments", "usage:",
        )
        lowered_output = output.lower()
        if returncode == -1 or any(marker in lowered_output for marker in infrastructure_markers):
            return {
                **summary,
                "status": "inconclusive",
                "reason": "The command failed because of setup/tooling rather than the described problem.",
            }

        signals = problem.get("failure_signals") or []
        if isinstance(signals, str):
            signals = [signals]
        matched = []
        for signal in signals:
            signal_text = str(signal).strip()
            if len(signal_text) >= 8 and signal_text.lower() in lowered_output:
                matched.append(signal_text[:300])
                continue
            for code in re.findall(r"\b[A-Z][A-Z0-9_-]*\d{2,4}\b", signal_text):
                if code.lower() in lowered_output:
                    matched.append(code)
        if matched:
            return {
                **summary,
                "status": "failure_confirmed",
                "reason": "Command output contains the candidate's recorded failure signal.",
                "matched_signals": list(dict.fromkeys(matched)),
            }
        return {
            **summary,
            "status": "inconclusive",
            "reason": "Command failed, but its output does not directly match the candidate failure.",
        }

    @staticmethod
    def _safe_lightweight_command(
        command: str,
        root: Path,
        resolved: List[Path],
    ) -> tuple[Optional[str], str]:
        if not command:
            return None, "Candidate provides no verification command."
        if len(command) > 500 or re.search(r"[;&|><`\n]|\$\(", command):
            return None, "Command is not a safe single lightweight invocation."
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None, "Command cannot be parsed safely."
        if not tokens:
            return None, "Candidate provides no verification command."

        executable = Path(tokens[0]).name.lower()
        tool = executable
        if executable in {"python", "python3"}:
            if len(tokens) < 3 or tokens[1] != "-m":
                return None, "Only Python module validation commands are allowed."
            tool = tokens[2].lower()
        if tool not in {"ruff", "mypy", "pyright", "pytest"}:
            return None, f"'{tool}' is not in the lightweight validation allowlist."
        mutating_flags = {"--fix", "--unsafe-fixes", "--install-types"}
        if any(token in mutating_flags for token in tokens):
            return None, "Potentially mutating validation flags are not allowed."
        if tool == "ruff" and "format" in tokens and "--check" not in tokens:
            return None, "ruff format is allowed only with --check."

        if tool == "pytest":
            has_specific_test = False
            for token in tokens:
                if not (token.endswith(".py") or "::" in token):
                    continue
                test_path = (root / token.split("::", 1)[0]).resolve()
                try:
                    test_path.relative_to(root)
                except ValueError:
                    continue
                if test_path.is_file():
                    has_specific_test = True
                    break
            if not has_specific_test:
                return None, "A repository-wide pytest invocation is not lightweight."

        prefix = "PYTHONDONTWRITEBYTECODE=1"
        if tool == "pytest":
            prefix += " PYTEST_ADDOPTS='-p no:cacheprovider'"
        suffix = ""
        if tool == "ruff" and "--no-cache" not in tokens:
            suffix = " --no-cache"
        elif tool == "mypy" and not any(
            token == "--cache-dir" or token.startswith("--cache-dir=")
            for token in tokens
        ):
            suffix = " --cache-dir=/dev/null"
        return f"{prefix} {command}{suffix}", ""

    @staticmethod
    def _is_observed_target_problem(problem: Dict[str, Any]) -> bool:
        source = problem.get("source")
        return (
            str(problem.get("problem_type", "")).lower() == "ci_failure"
            or str(problem.get("evidence_status", "")).lower() == "observed_in_target_ci"
            or (isinstance(source, dict) and bool(source.get("current_failure")))
        )

    @staticmethod
    def _candidate_files(problem: Dict[str, Any]) -> List[str]:
        files = problem.get("files") or problem.get("relevant_files") or []
        if isinstance(files, str):
            files = [files]
        normalized = []
        for item in files:
            if isinstance(item, dict):
                item = item.get("file") or item.get("path") or ""
            value = str(item).strip()
            if value:
                normalized.append(value)
        return normalized

    @staticmethod
    def _resolve_checkout_file(root: Path, claimed: str) -> Optional[Path]:
        """Resolve a memory path without permitting access outside checkout."""
        clean = claimed.strip().replace("\\", "/")
        clean = re.sub(r"^[ab]/", "", clean)
        parts = [part for part in clean.split("/") if part not in ("", ".")]
        if not parts or ".." in parts or Path(clean).is_absolute():
            return None

        candidate = (root / Path(*parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate

        # Suffix matching handles memory paths such as ``agno/foo.py`` versus
        # checkout ``libs/agno/agno/foo.py``.
        basename = parts[-1]
        matches = []
        for candidate in root.rglob(basename):
            if not candidate.is_file():
                continue
            candidate = candidate.resolve()
            try:
                relative_parts = candidate.relative_to(root).parts
            except ValueError:
                continue
            if any(part in {".git", ".venv", "venv", "node_modules"} for part in relative_parts):
                continue
            common_suffix = 0
            for left, right in zip(reversed(parts), reversed(relative_parts)):
                if left != right:
                    break
                common_suffix += 1
            matches.append((common_suffix, len(relative_parts), candidate))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[1], str(item[2])))
        # A basename-only match is safe only when unique; otherwise the location
        # is ambiguous and must not be promoted to the repair agent.
        best_score = matches[0][0]
        best = [item for item in matches if item[0] == best_score]
        return best[0][2] if best_score > 1 or len(best) == 1 else None

    @staticmethod
    def _read_source_evidence(
        root: Path,
        paths: List[Path],
        problem: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        terms = re.findall(
            r"[A-Za-z_][A-Za-z0-9_]{3,}",
            " ".join(str(problem.get(key, "")) for key in ("problem", "root_cause")),
        )
        evidence: List[Dict[str, str]] = []
        total = 0
        for path in paths[:5]:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            limit = min(12000, 30000 - total)
            if limit <= 0:
                break
            if len(content) > limit:
                positions = [content.lower().find(term.lower()) for term in terms]
                positions = [position for position in positions if position >= 0]
                center = positions[0] if positions else 0
                start = max(0, center - limit // 3)
                excerpt = content[start:start + limit]
            else:
                excerpt = content
            total += len(excerpt)
            evidence.append({
                "path": str(path.relative_to(root)),
                "excerpt": excerpt,
            })
        return evidence

    def _review_candidate_source(
        self,
        problem: Dict[str, Any],
        evidence: List[Dict[str, str]],
        index: int,
        command_check: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        candidate = {
            key: problem.get(key)
            for key in (
                "problem", "root_cause", "problem_type", "failure_type",
                "failure_signals", "files", "repair_strategy",
            )
        }
        prompt = f"""Act as a conservative, read-only source validation agent.

Determine whether the FAILURE-STATE source excerpts directly contain the defect or
necessary faulty precondition described by this retrieved memory candidate.

Candidate {index}:
{json.dumps(candidate, indent=2, default=str)}

Failure-state source excerpts:
{json.dumps(evidence, indent=2, default=str)}

Lightweight verification result (may be skipped or inconclusive):
{json.dumps(command_check or {}, indent=2, default=str)}

Rules:
- "confirmed" requires concrete source evidence of the described defect/precondition.
- File existence, similar technology, or a plausible repair alone is insufficient.
- Use "rejected" when the source contradicts the candidate or already contains the fix.
- Use "unverifiable" when runtime behavior, missing context, or ambiguity prevents proof.
- Do not propose or apply a repair.

Return exactly one JSON object:
{{
  "decision": "confirmed|rejected|unverifiable",
  "reason": "brief evidence-based explanation",
  "evidence": ["path: line/construct supporting the decision"]
}}

{STRICT_JSON_RULES}
"""
        try:
            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt,
                parse_json=True,
                max_tokens=1200,
                verbose=False,
            )
        except Exception as exc:
            return {
                "decision": "unverifiable",
                "reason": f"Source validation failed: {type(exc).__name__}",
                "evidence": [],
            }
        if not isinstance(response, dict):
            return {
                "decision": "unverifiable",
                "reason": "Source validator returned malformed output.",
                "evidence": [],
            }
        decision = str(response.get("decision", "")).strip().lower()
        if decision not in {"confirmed", "rejected", "unverifiable"}:
            decision = "unverifiable"
        evidence_items = response.get("evidence") or []
        if isinstance(evidence_items, str):
            evidence_items = [evidence_items]
        return {
            "decision": decision,
            "reason": str(response.get("reason") or "No validation reason supplied."),
            "evidence": [str(item) for item in evidence_items[:8]],
        }

    @staticmethod
    def _with_source_decision(
        problem: Dict[str, Any],
        decision: str,
        reason: str,
        evidence: List[str],
    ) -> Dict[str, Any]:
        annotated = dict(problem)
        annotated["source_validation"] = {
            "decision": decision,
            "reason": reason,
            "evidence": evidence,
        }
        return annotated

    def decompose_only(
        self,
        ci_failure: Dict[str, Any],
        verification: Optional[Dict[str, Any]] = None,
        issue_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decompose CI failure into problems without memory retrieval.
        Used for baseline mode where only decomposition is needed.

        Args:
            ci_failure: Complete CI failure analysis dict
            verification: Workflow verification dict
            issue_metadata: Metadata (workflow_path, workflow_name, repo, sha_fail)

        Returns:
            List of decomposed problems (without memory data)
        """
        issue_metadata = issue_metadata or {}
        sha_fail = issue_metadata.get("sha_fail", "")

        # Check cache first
        cache = get_global_cache()
        if sha_fail and cache.has(sha_fail):
            decomposed_problems = cache.get_problems(sha_fail)
            print(f"[Memory] Using cached decomposition for {sha_fail[:12]}")
            return decomposed_problems

        # Generate decomposition and save to cache
        print(f"[Memory] Generating decomposition for {sha_fail[:12] if sha_fail else 'unknown'}")
        return self._decompose_and_save_to_cache(
            ci_failure=ci_failure,
            verification=verification,
            workflow_path=issue_metadata.get("workflow_path", ""),
            workflow_name=issue_metadata.get("workflow_name", ""),
            repo=issue_metadata.get("repo", ""),
            sha_fail=sha_fail
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

**CRITICAL: Only generate well-defined problems**
- Problem description must be specific and clear (not vague like "job failed without details")
- Root cause must be technically specific (not vague like "insufficient log information")
- Failure signals must be concrete error messages from CI logs (not placeholders)
- Files should be specified when possible (can be empty only if problem/root_cause/signals are all well-defined)
- If problem, root_cause, or failure_signals are vague or unclear, return {{"problems": []}} instead

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

        # Track API call timing
        import time
        start_time = time.time()

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)

        duration = time.time() - start_time

        # Record to metrics if available
        if self.metrics_recorder:
            from utilities.run_metrics import extract_usage, response_cost_usd, estimate_cost_usd
            usage = extract_usage(response)
            cost, source = response_cost_usd(response)
            if cost == 0.0 and source == "unavailable":
                # Estimate cost from usage
                model_name = getattr(self.llm, "model_name", "") or getattr(self.llm, "model", "")
                cost, source = estimate_cost_usd(str(model_name), usage)

            self.metrics_recorder.record_api_call(
                phase="memory.decomposition",
                model=getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None),
                duration_seconds=duration,
                usage=usage,
                cost_usd=cost,
                cost_source=source
            )

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

        failed_jobs = (
            ci_failure.get("failed_jobs")
            or ci_failure.get("failed_job")
            or []
        )
        # Prefer commands that actually failed. Fall back to the workflow
        # validation sequence only for older analyzer records.
        failed_cmd = [
            job.get("command")
            for job in failed_jobs
            if isinstance(job, dict) and job.get("command")
        ] or self._extract_failed_commands(verification)

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
            "failed_jobs": failed_jobs,

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
            "- **CRITICAL**: After fixing each problem, re-read any modified files before starting the next problem",
            "- **CRITICAL**: Check if the previous fix already resolved the current problem - if yes, skip it",
            "- Follow the action steps precisely",
            "- Run validation command after each fix",
            "- If validation fails due to setup issues, document and continue",
            "- If new problems arise during fixes, address them as well",
            "",
            "**IMPORTANT**: Multiple problems may affect the same file/lines. Always verify the current",
            "state before making changes to avoid creating duplicate fixes!",
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
