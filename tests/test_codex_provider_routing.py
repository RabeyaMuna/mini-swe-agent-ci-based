from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from codex.scripts.run_codex_ci_repair import existing_prediction_ids  # noqa: E402


LAUNCHER = PROJECT_ROOT / "run_codex_direct.sh"
MINISWE_LAUNCHER = PROJECT_ROOT / "run_miniswe_direct.sh"


def _configure(tmp_path: Path, model: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "MINIMAX_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "CODEX_PROVIDER",
        "CODEX_API_BASE",
    ):
        env.pop(name, None)
    env.update(
        {
            "OPENAI_API_KEY": "sk-openai-test-only",
            "OPENROUTER_API_KEY": "sk-openrouter-test-only",
            "CODEX_CONFIG_ONLY": "1",
        }
    )
    return subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "1",
            "baseline",
            "backward",
            model,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_openai_and_minimax_use_isolated_provider_configs(tmp_path: Path) -> None:
    openai_result = _configure(tmp_path, "openai/gpt-5.4-mini")
    assert openai_result.returncode == 0, openai_result.stdout + openai_result.stderr
    assert "provider=OpenAI" in openai_result.stdout
    assert "endpoint=https://api.openai.com/v1" in openai_result.stdout

    openai_home = tmp_path / ".codex-local" / "gpt-5_4-mini"
    openai_config_before = (openai_home / "config.toml").read_text()
    openai_auth = (openai_home / "auth.json").read_text()
    assert "model_provider" not in openai_config_before
    assert "sk-openai-test-only" in openai_auth
    assert "sk-openrouter-test-only" not in openai_auth

    minimax_result = _configure(tmp_path, "minimax2.5")
    assert minimax_result.returncode == 0, minimax_result.stdout + minimax_result.stderr
    assert "Model:     minimax/minimax-m2.5" in minimax_result.stdout
    assert "provider=OpenRouter" in minimax_result.stdout
    assert "endpoint=https://openrouter.ai/api/v1" in minimax_result.stdout

    minimax_home = tmp_path / ".codex-local" / "minimax_minimax-m2_5"
    minimax_config = (minimax_home / "config.toml").read_text()
    minimax_auth = (minimax_home / "auth.json").read_text()
    assert 'model_provider = "openrouter"' in minimax_config
    assert 'base_url = "https://openrouter.ai/api/v1"' in minimax_config
    assert "sk-openrouter-test-only" in minimax_auth
    assert "sk-openai-test-only" not in minimax_auth

    # Configuring MiniMax must never overwrite the existing OpenAI config.
    assert (openai_home / "config.toml").read_text() == openai_config_before


def _configure_miniswe(tmp_path: Path, model: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
    ):
        env.pop(name, None)
    env.update(
        {
            "OPENAI_API_KEY": "sk-openai-test-only",
            "OPENROUTER_API_KEY": "sk-openrouter-test-only",
            "MINISWE_CONFIG_ONLY": "1",
        }
    )
    return subprocess.run(
        [
            "bash",
            str(MINISWE_LAUNCHER),
            "1",
            "baseline",
            "backward",
            model,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_miniswe_routes_gpt_and_minimax_to_their_required_providers(
    tmp_path: Path,
) -> None:
    openai_result = _configure_miniswe(tmp_path, "gpt-5.4-mini")
    assert openai_result.returncode == 0, openai_result.stdout + openai_result.stderr
    assert "Provider:  OpenAI" in openai_result.stdout
    assert "Endpoint:  https://api.openai.com/v1" in openai_result.stdout

    minimax_result = _configure_miniswe(tmp_path, "minimax2.5")
    assert minimax_result.returncode == 0, minimax_result.stdout + minimax_result.stderr
    assert "Provider:  OpenRouter" in minimax_result.stdout
    assert "Endpoint:  https://openrouter.ai/api/v1" in minimax_result.stdout


def test_codex_resume_reads_only_the_matching_prediction_file(tmp_path: Path) -> None:
    results_root = tmp_path / "results" / "codex"
    predictions_dir = results_root / "baseline_gpt-5_4-mini"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "predictions.json").write_text(
        '[{"id": 1}, {"id": "43"}]', encoding="utf-8"
    )

    assert existing_prediction_ids(
        results_root, "baseline", "gpt-5.4-mini"
    ) == {"1", "43"}
    assert existing_prediction_ids(
        results_root, "baseline", "minimax/minimax-m2.5"
    ) == set()
