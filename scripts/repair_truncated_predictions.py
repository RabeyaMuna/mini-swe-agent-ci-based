"""Repair Git patches truncated by the runner's former ``.strip()`` call."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


def _last_hunk_deficit(diff: str) -> int:
    """Return trailing blank context records removed from the final hunk."""
    lines = diff.splitlines()
    hunks: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        match = HUNK_HEADER.match(lines[index])
        if match is None:
            index += 1
            continue

        expected_old = int(match.group(2) or 1)
        expected_new = int(match.group(4) or 1)
        actual_old = 0
        actual_new = 0
        index += 1
        while index < len(lines) and not lines[index].startswith(
            ("diff --git ", "@@ ")
        ):
            line = lines[index]
            if line.startswith(" "):
                actual_old += 1
                actual_new += 1
            elif line.startswith("-"):
                actual_old += 1
            elif line.startswith("+"):
                actual_new += 1
            elif line.startswith("\\ No newline at end of file"):
                pass
            else:
                break
            index += 1
        hunks.append((expected_old - actual_old, expected_new - actual_new))

    if not hunks:
        return 0
    if any(old or new for old, new in hunks[:-1]):
        raise ValueError("A non-final hunk has inconsistent line counts")

    old_deficit, new_deficit = hunks[-1]
    if old_deficit < 0 or new_deficit < 0 or old_deficit != new_deficit:
        raise ValueError(
            "Final hunk cannot be repaired with trailing blank context records: "
            f"old deficit={old_deficit}, new deficit={new_deficit}"
        )
    return old_deficit


def patch_syntax_error(diff: str) -> str | None:
    """Return Git's parse error, or ``None`` when the patch is valid."""
    result = subprocess.run(
        ["git", "apply", "--numstat", "-"],
        input=diff.encode(),
        capture_output=True,
        check=False,
    )
    if not result.returncode:
        return None
    return result.stderr.decode(errors="replace").strip()


def validate_patch_syntax(diff: str) -> None:
    """Require Git to parse the patch independently of a repository checkout."""
    error = patch_syntax_error(diff)
    if error is not None:
        raise ValueError(f"Git rejected repaired patch syntax: {error}")


def repair_truncated_diff(diff: str) -> tuple[str, int]:
    """Restore the exact trailing records that ``str.strip()`` removed."""
    if not diff:
        raise ValueError("Cannot repair an empty patch")

    deficit = _last_hunk_deficit(diff)
    repaired = diff if diff.endswith("\n") else f"{diff}\n"
    repaired += " \n" * deficit
    validate_patch_syntax(repaired)
    return repaired, deficit


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def repair_predictions(
    predictions_path: Path,
    report_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise TypeError(f"Expected a JSON list in {predictions_path}")

    prediction_by_id = {
        str(item.get("id")): item
        for item in predictions
        if isinstance(item, dict) and item.get("id") is not None
    }
    if report_path is not None:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "corrupted_patches" in report:
            report_items = report["corrupted_patches"]
        else:
            report_items = report.get("unable_to_apply") or []
        target_ids = {str(item["id"]) for item in report_items}
        missing_ids = sorted(target_ids - prediction_by_id.keys())
        if missing_ids:
            raise ValueError(f"Report IDs missing from predictions: {missing_ids}")
    else:
        target_ids = {
            prediction_id
            for prediction_id, item in prediction_by_id.items()
            if item.get("diff") and patch_syntax_error(str(item["diff"])) is not None
        }

    deficits: Counter[int] = Counter()
    changed_ids: list[str] = []
    already_valid_ids: list[str] = []
    for prediction_id in sorted(target_ids):
        prediction = prediction_by_id[prediction_id]
        diff = str(prediction.get("diff") or "")
        if patch_syntax_error(diff) is None:
            already_valid_ids.append(prediction_id)
            continue
        repaired, deficit = repair_truncated_diff(diff)
        deficits[deficit] += 1
        prediction["diff"] = repaired
        if "patch_bytes" in prediction:
            prediction["patch_bytes"] = len(repaired.encode())
        changed_ids.append(prediction_id)

    if not dry_run and changed_ids:
        _atomic_write_json(predictions_path, predictions)

    return {
        "selected": len(target_ids),
        "repaired": len(changed_ids),
        "already_valid": len(already_valid_ids),
        "trailing_context_deficits": dict(sorted(deficits.items())),
        "selection": "report" if report_path is not None else "git-syntax-scan",
        "dry_run": dry_run,
        "written": bool(changed_ids) and not dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        help="Optional prior corruption report; otherwise scan every patch with Git",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = repair_predictions(args.predictions, args.report, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
