"""Shared model aliases for CI-Repair-Bench commands.

The codebase passes model names through LiteLLM. For OpenRouter-hosted models,
LiteLLM expects the provider prefix: ``openrouter/<openrouter-model-slug>``.
"""

from __future__ import annotations

import os
import re


DEFAULT_MINIMAX_MODEL = "openrouter/minimax/minimax-m2.5"
DEFAULT_GLM_MODEL = "openrouter/z-ai/glm-5.2"


_ALIASES = {
    "minimax": DEFAULT_MINIMAX_MODEL,
    "minimax2.5": DEFAULT_MINIMAX_MODEL,
    "minimax-2.5": DEFAULT_MINIMAX_MODEL,
    "minimaxm2.5": DEFAULT_MINIMAX_MODEL,
    "minimax-m2.5": DEFAULT_MINIMAX_MODEL,
    "minimax_m2.5": DEFAULT_MINIMAX_MODEL,
    "minimax/minimax-m2.5": DEFAULT_MINIMAX_MODEL,
    "openrouter/minimax/minimax-m2.5": DEFAULT_MINIMAX_MODEL,
    "glm": DEFAULT_GLM_MODEL,
    "glm5.2": DEFAULT_GLM_MODEL,
    "glm-5.2": DEFAULT_GLM_MODEL,
    "glm_5.2": DEFAULT_GLM_MODEL,
    "glm_5_2": DEFAULT_GLM_MODEL,
    "glm-5-2": DEFAULT_GLM_MODEL,
    "z-ai/glm-5.2": DEFAULT_GLM_MODEL,
    "openrouter/z-ai/glm-5.2": DEFAULT_GLM_MODEL,
    # Keep the old GLM value valid for existing experiments.
    "glm-4-plus": "glm-4-plus",
}


def resolve_model_alias(model_name: str | None) -> str | None:
    """Resolve a user-facing model name to the provider string used by LiteLLM."""
    if model_name is None:
        return None

    raw = str(model_name).strip()
    if not raw:
        return raw

    lowered = raw.lower()

    if lowered in _ALIASES:
        return _ALIASES[lowered]

    if lowered.startswith("minimax/"):
        return f"openrouter/{raw}"

    if lowered.startswith("z-ai/"):
        return f"openrouter/{raw}"

    return raw


def model_output_name(model_name: str | None) -> str:
    """Return a stable, filesystem-friendly folder name for result paths."""
    resolved = resolve_model_alias(model_name) or "unknown-model"

    if resolved == DEFAULT_MINIMAX_MODEL:
        return "minimax-m2.5"
    if resolved == DEFAULT_GLM_MODEL:
        return "glm-5.2"
    if resolved == "glm-4-plus":
        return "glm-4-plus"

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
