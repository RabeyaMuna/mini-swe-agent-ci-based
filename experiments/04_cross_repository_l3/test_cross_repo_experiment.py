from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_miniswe_ci_bench import _ablation_to_miniswe  # noqa: E402


SCRIPT_PATH = Path(__file__).with_name("prepare_cross_repo_data.py")
SPEC = importlib.util.spec_from_file_location("prepare_cross_repo_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_cross_repo_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_cross_repo_data
SPEC.loader.exec_module(prepare_cross_repo_data)


def test_single_level_ablations_are_isolated() -> None:
    assert _ablation_to_miniswe("L1") == (True, "L1")
    assert _ablation_to_miniswe("L2") == (True, "L2")
    assert _ablation_to_miniswe("L3") == (True, "L3")


def test_unknown_ablation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported ablation"):
        _ablation_to_miniswe("L4")


def test_cross_repo_data_is_disjoint_and_lineage_consistent() -> None:
    data = prepare_cross_repo_data.build_cross_repo_data(
        prepare_cross_repo_data.DEFAULT_EVAL_DATASET
    )

    assert len(data.eval_ids) == 100
    assert len(data.target_repos) == 30
    assert len(data.source_repos) == 45
    assert data.source_repos.isdisjoint(data.target_repos)
    assert len(data.l1) == 110
    assert len(data.l2) == 110
    assert len(data.l3) == 302

    retained_ids = set(data.retained_ids)
    assert {str(record["issue_id"]) for record in data.l2} == retained_ids
    assert {str(record["source_issue_id"]) for record in data.l3} == retained_ids
