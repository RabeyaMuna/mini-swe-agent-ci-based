from __future__ import annotations

from types import SimpleNamespace

from utilities.llm_model import LitellmModel


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
