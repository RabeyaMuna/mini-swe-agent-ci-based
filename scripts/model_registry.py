"""Shared model aliases for CI-Repair-Bench commands.

The codebase passes model names through LiteLLM. For OpenRouter-hosted models,
LiteLLM expects the provider prefix: ``openrouter/<openrouter-model-slug>``.
"""

from __future__ import annotations

import os
import re


CANONICAL_MINIMAX_MODEL = "openrouter/minimax/minimax-m2.5"
CANONICAL_ZAI_GLM_MODEL = "zai/glm-5.2"
LEGACY_GLM_MODEL = "glm-4-plus"

MINIMAX_ALIASES = {
    "minimax",
    "minimax2.5",
    "minimax-2.5",
    "minimaxm2.5",
    "minimax-m2.5",
    "minimax_m2.5",
}

GLM_ALIASES = {
    "glm",
    "gml",
    "gml5.2",
    "gml-5.2",
    "glm5.2",
    "glm-5.2",
    "glm_5.2",
    "glm_5_2",
    "glm-5-2",
}


_ALIASES = {
    **{alias: CANONICAL_MINIMAX_MODEL for alias in MINIMAX_ALIASES},
    "minimax/minimax-m2.5": CANONICAL_MINIMAX_MODEL,
    "openrouter/minimax/minimax-m2.5": CANONICAL_MINIMAX_MODEL,
    **{alias: CANONICAL_ZAI_GLM_MODEL for alias in GLM_ALIASES},
    "z-ai/glm-5.2": CANONICAL_ZAI_GLM_MODEL,
    "zai/glm-5.2": CANONICAL_ZAI_GLM_MODEL,
    "openrouter/z-ai/glm-5.2": CANONICAL_ZAI_GLM_MODEL,
    # Keep the old GLM value valid for existing experiments.
    LEGACY_GLM_MODEL: LEGACY_GLM_MODEL,
}


def resolve_model_alias(model_name: str | None) -> str | None:
    """Resolve a user-facing model name to the provider string used by LiteLLM."""
    if model_name is None:
        return None

    raw = str(model_name).strip()
    if not raw:
        return raw

    lowered = raw.lower()

    if lowered in GLM_ALIASES:
        if os.getenv("GLM_MODEL_NAME"):
            return os.getenv("GLM_MODEL_NAME", "").strip()
        return CANONICAL_ZAI_GLM_MODEL

    if lowered in MINIMAX_ALIASES and os.getenv("MINIMAX_MODEL_NAME"):
        return os.getenv("MINIMAX_MODEL_NAME", "").strip()

    if lowered in _ALIASES:
        return _ALIASES[lowered]

    if lowered.startswith("minimax/"):
        return f"openrouter/{raw}"

    if lowered.startswith(("z-ai/", "zai/")):
        return raw

    return raw


def model_output_name(model_name: str | None) -> str:
    """Return a stable, filesystem-friendly folder name for result paths."""
    resolved = resolve_model_alias(model_name) or "unknown-model"

    if resolved == CANONICAL_MINIMAX_MODEL:
        return "minimax-m2.5"
    if resolved == CANONICAL_ZAI_GLM_MODEL:
        return "glm-5.2"
    if resolved == LEGACY_GLM_MODEL:
        return LEGACY_GLM_MODEL

    name = resolved
    if name.startswith("openrouter/"):
        name = name.removeprefix("openrouter/")
    name = name.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "model"


def configure_model_environment(model_name: str | None) -> str | None:
    """Resolve a model and expose it to memory/decomposition helpers."""
    resolved = resolve_model_alias(model_name)
    if resolved:
        os.environ["MEMCI_LLM_MODEL"] = resolved
    return resolved
