"""Minimal repair session - tracks problems and validates fixes."""

import json
from pathlib import Path
from typing import Callable

from minisweagent.run.benchmarks.utils.git_patch import (
    collect_workspace_patch, conflicted_paths, check_resolved_files,
)


def validation_commands(problem: dict) -> list[str]:
    """Extract validation commands from problem definition."""
    commands = []
    sources = [problem]
    for key in ("verification", "repair_strategy"):
        if isinstance(problem.get(key), dict):
            sources.append(problem[key])
    for source in list(sources):
        sequence = source.get("validation_sequence", [])
        if isinstance(sequence, list):
            sources.extend(step for step in sequence if isinstance(step, dict))
    for source in sources:
        command = source.get("verification_cmd") or source.get("validation_cmd")
        if isinstance(command, str) and command.strip().lower() not in ("", "n/a", "none"):
            commands.append(command.strip())
    return list(dict.fromkeys(commands))


class RepairSession:
    """Minimal session - just tracks problems and checkpoints."""

    def __init__(self, checkout: Path, sha_fail: str, problems: list[dict], artifacts: Path):
        self.checkout = checkout
        self.sha_fail = sha_fail
        self.problems = problems
        import uuid
        self.artifacts = artifacts / f"attempt-{uuid.uuid4().hex}"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.validation_history = []
        self.known_conflicts = set()
        self.latest_checkpoint = None
        self.record_path = self.artifacts / "repair-record.json"
        self.save()
        self.checkpoint("initial")

    def save(self):
        """Save current state."""
        self.record_path.write_text(json.dumps({
            "sha_fail": self.sha_fail,
            "problems": self.problems,
            "repairs": self.records,
            "validation_history": self.validation_history,
        }, indent=2, default=str), encoding="utf-8")

    def checkpoint(self, label: str) -> dict:
        """Save current workspace state."""
        try:
            patch = collect_workspace_patch(self.checkout, self.sha_fail)
            checkpoint_file = self.artifacts / f"{label}.patch"
            checkpoint_file.write_text(patch, encoding="utf-8")
            self.latest_checkpoint = str(checkpoint_file)
            from minisweagent.run.benchmarks.utils.patch_merger import parse_unified_diff
            files = [p['file'] for p in parse_unified_diff(patch)] if patch else []
            return {"checkpoint": str(checkpoint_file), "changed_files": files}
        except Exception:
            return {"checkpoint": None, "changed_files": []}

    def validate(self, indices: list[int], execute: Callable | None, label: str) -> dict:
        """Run validation commands - simplified, no scoping."""
        results = {}
        problems = []

        for index in indices:
            commands = validation_commands(self.problems[index])
            for command in commands:
                if command in results:
                    continue

                result = {"command": command, "returncode": None, "status": "unavailable"}
                output = "No command executor is available."
                if execute is not None:
                    try:
                        response = execute(command)
                        result["returncode"] = response.get("returncode")
                        output = str(response.get("output", "")) + str(response.get("exception_info", ""))
                        result["status"] = (
                            "passed" if result["returncode"] == 0 else
                            "unavailable" if result["returncode"] in (None, 126, 127, -1) else "failed"
                        )
                    except Exception as exc:
                        output = f"Validation could not execute: {exc}"
                log = self.artifacts / f"{label}-check-{len(results) + 1}.log"
                log.write_text(output, encoding="utf-8", errors="replace")
                result["output_path"] = str(log)
                results[command] = result

            statuses = [results[command]["status"] for command in commands]
            status = "failed" if "failed" in statuses else (
                "passed" if statuses and all(s == "passed" for s in statuses) else "unverified"
            )
            problems.append({"problem_id": index + 1, "commands": commands, "status": status})

        status = "failed" if any(p["status"] == "failed" for p in problems) else (
            "passed" if problems and all(p["status"] == "passed" for p in problems) else "unverified"
        )
        report = {"label": label, "status": status, "problems": problems, "checks": list(results.values())}
        self.validation_history.append(report)
        self.save()
        return report

    def record(self, index: int, info: dict, execute: Callable | None):
        """Record agent result for one problem."""
        validation = self.validate([index], execute, f"problem-{index + 1}")
        entry = {
            "problem_id": index + 1,
            "agent_exit_status": info.get("exit_status"),
            "submission": info.get("submission", ""),
            "error": info.get("error"),
            "validation": validation,
            **self.checkpoint(f"problem-{index + 1}")
        }
        self.records.append(entry)
        self.save()
        return entry

    def context(self) -> str:
        """Return context for agent prompts."""
        if not self.records:
            return ""
        recent = [{"problem": i + 1, "status": r.get("agent_exit_status")}
                  for i, r in enumerate(self.records[-3:])]
        return (
            f"\n\nPrevious repairs in this session:\n"
            f"{json.dumps(recent, indent=2)}\n"
            "Preserve earlier repairs unless current evidence proves they conflict.\n"
        )

    def finish(self, execute: Callable | None, repair: Callable | None, max_attempts: int = 2):
        """Collect final patch and validate - simplified, no reconciliation."""
        from minisweagent.run.benchmarks.utils.git_patch import UnresolvedMergeConflict

        # Check for Git conflicts
        self.known_conflicts.update(conflicted_paths(self.checkout))
        try:
            check_resolved_files(self.checkout, list(self.known_conflicts))
            patch = collect_workspace_patch(self.checkout, self.sha_fail)
        except UnresolvedMergeConflict as exc:
            # Real Git conflict - cannot proceed
            raise

        # Validate (but don't trigger reconciliation on failure)
        report = self.validate(list(range(len(self.problems))), execute, "final-0")
        (self.artifacts / "candidate.patch").write_bytes(patch.encode("utf-8"))

        return patch, report
