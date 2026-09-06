"""Return actual merge conflicts to the repair agent before exporting a patch."""

import json
from pathlib import Path
from typing import Callable

from minisweagent.run.benchmarks.utils.git_patch import (
    UnresolvedMergeConflict, collect_workspace_patch, conflicted_paths, check_resolved_files,
)


def collect_with_reconciliation(
    checkout: Path,
    sha_fail: str,
    resolve_conflict: Callable[[str, int], None] | None = None,
    max_attempts: int = 2,
) -> str:
    """
    Collect workspace patch, only retrying for ACTUAL Git merge conflicts.

    SIMPLIFIED: Only handles real Git conflicts (unmerged index entries),
    not validation failures or pre-existing code issues.
    """
    known_conflicts = set()

    # First attempt: check for any existing conflicts
    known_conflicts.update(conflicted_paths(checkout))

    # If no Git conflicts exist, collect and return patch immediately
    if not known_conflicts:
        return collect_workspace_patch(checkout, sha_fail)

    # Git conflicts exist - try to resolve them
    for attempt in range(max_attempts + 1):
        try:
            check_resolved_files(checkout, list(known_conflicts))
            return collect_workspace_patch(checkout, sha_fail)
        except UnresolvedMergeConflict as exc:
            # Only invoke agent if we have a conflict resolver and attempts remaining
            if resolve_conflict is None or attempt == max_attempts:
                raise

            # This is a REAL Git conflict that needs agent intervention
            resolve_conflict(str(exc), attempt + 1)

            # Update known conflicts after resolution attempt
            known_conflicts.update(conflicted_paths(checkout))

    raise ValueError("max_attempts must be nonnegative")


def write_reconciliation_prompt(
    artifact_dir: Path,
    checkout: Path,
    sha_fail: str,
    problems: list,
    error: str,
    attempt: int,
    verification: dict | None = None,
) -> Path:
    """Keep full problem context on disk instead of truncating or inlining it."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context = artifact_dir / "problems.json"
    context.write_text(json.dumps({
        "sha_fail": sha_fail, "problems": problems, "verification": verification,
    }, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    prompt = artifact_dir / f"resolve-conflicts-{attempt}.md"
    prompt.write_text(f"""Reconcile the accumulated CI repairs and resolve the reported failures.

Checkout: {checkout}
Failed commit (the final patch base): {sha_fail}
Original problems and validation context: {context}
Integration failure: {error[:2000]}

Each problem was repaired sequentially in this SAME checkout. Read the problem
context file and inspect the current files, git status, and git ls-files -u.
Read large context files in bounded sections as needed. For actual Git conflicts,
inspect the base, ours, and theirs index stages to understand both changes.
Validation regressions may have no Git conflict: use the earlier requirements,
checkpoints and failing check output to repair the combined behavior.

Resolve conflicts by preserving the intended fixes for ALL supplied problems.
Edit the source files directly and stage only the resolved paths with git add
or git rm, as appropriate, so no unmerged index entries remain. Do not reset the
checkout, abort the merge, discard earlier repairs, or choose an entire side
without reconciling the other side's required behavior. Keep existing repository
constraints and use the repository's configured environment.

Run the supplied validation commands for all affected problems against the
combined final state. Correct regressions introduced by the resolution. Avoid
unrelated changes or formatting. Report any checks you cannot run or failures
you cannot resolve; do not claim those checks passed.

Do not construct, concatenate, trim, or reformat patch text. Leave the resolved
files in this checkout. The runner will generate one Git diff against the failed
commit and independently verify that applying it reproduces the final Git tree.
""", encoding="utf-8")
    return prompt
