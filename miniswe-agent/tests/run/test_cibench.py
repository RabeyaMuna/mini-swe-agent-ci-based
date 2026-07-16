from types import SimpleNamespace
from unittest.mock import patch

from minisweagent.run.benchmarks.cibench import _make_context_llm, _resolve_context_model


def test_context_model_defaults_to_repair_model():
    config = {"model": {"model_name": "minimax/minimax-m2.5"}}

    assert _resolve_context_model(None, config) == "minimax/minimax-m2.5"


def test_context_model_can_be_overridden():
    config = {"model": {"model_name": "minimax/minimax-m2.5"}}

    assert _resolve_context_model("openai/gpt-4.1", config) == "openai/gpt-4.1"


def test_context_llm_uses_configured_model():
    config = {
        "model": {
            "model_name": "minimax/minimax-m2.5",
            "model_class": "openrouter",
            "model_kwargs": {"temperature": 0.0, "drop_params": True},
        }
    }
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer"),
            )
        ]
    )

    with patch("minisweagent.run.benchmarks.cibench.litellm.completion", return_value=response) as completion:
        context_llm = _make_context_llm(config)
        assert context_llm("prompt") == "answer"

    completion.assert_called_once_with(
        model="minimax/minimax-m2.5",
        messages=[{"role": "user", "content": "prompt"}],
        temperature=0.0,
        drop_params=True,
    )
