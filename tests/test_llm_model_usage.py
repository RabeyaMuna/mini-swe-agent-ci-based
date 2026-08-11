from __future__ import annotations

from types import SimpleNamespace

from utilities.llm_model import LitellmModel


def test_gpt_credentials_use_openai_when_both_provider_keys_exist(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test-only")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    model = LitellmModel("gpt-5.4-mini")

    assert model.api_key == "sk-openai-test-only"
    assert model.api_base == "https://api.openai.com/v1"


def test_explicit_openrouter_model_uses_openrouter_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-only")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test-only")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    model = LitellmModel("openrouter/minimax/minimax-m2.5")

    assert model.api_key == "sk-openrouter-test-only"
    assert model.api_base == "https://openrouter.ai/api/v1"


def test_unknown_model_does_not_default_to_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test-only")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    model = LitellmModel("unconfigured-provider/model")

    assert model.api_key is None
    assert model.api_base is None


def test_invoke_reports_cached_input_tokens(monkeypatch, capsys) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok"),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=2000,
            completion_tokens=100,
            total_tokens=2100,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=1536,
                cache_write_tokens=None,
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=64),
        ),
    )
    monkeypatch.setattr(
        "utilities.llm_model.litellm.completion", lambda **_kwargs: response
    )

    result = LitellmModel("gpt-5.4-mini").invoke("repair this issue", max_tokens=100)

    assert result.content == "ok"
    assert "cached_input=1536" in capsys.readouterr().out
