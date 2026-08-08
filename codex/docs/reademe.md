# Running MiniMax M2.5 and Other Non‑OpenAI Models with Codex CLI

This guide explains exactly how this project wires non‑OpenAI models (like MiniMax M2.5) into OpenAI Codex CLI, alongside OpenAI models (GPT‑5‑mini, GPT‑5.4‑mini). It covers provider configuration, environment variables, model naming, and how the wrappers here make it work end‑to‑end.

## What Codex CLI Reads

Codex CLI reads config and credentials from `CODEX_HOME` (defaults to `~/.codex`). Two files matter:

- `config.toml` – provider and runtime settings
- `auth.json` – authentication (API key(s))

You can use either a project‑local config (e.g., `codex/.codex-config/`) or the user config at `~/.codex`. In this repo we standardize on `~/.codex` to avoid committing secrets.

## Providers and Model Names

Codex supports multiple providers:
- OpenAI (native): use OpenAI model names directly, e.g. `gpt-5-mini`, `gpt-5.4-mini-2026-03-17`.
- OpenRouter (OpenAI‑compatible): use slugs with provider prefixes, e.g. `minimax/minimax-m2.5`, `openai/gpt-5-mini`.

Key differences:
- OpenAI native does not require a `model_provider` entry; only `auth.json` with `OPENAI_API_KEY`.
- OpenRouter requires `model_provider = "openrouter"` and the OpenRouter API key placed in `OPENAI_API_KEY` (Codex expects an "OpenAI‑compatible" key name).

## Recommended Setup (used by this repo)

We run MiniMax M2.5 via OpenRouter and run GPT‑5 models directly via OpenAI. The wrappers write the correct provider config to `~/.codex` automatically before invoking Codex.

### A) OpenAI models (GPT‑5‑mini, GPT‑5.4‑mini‑2026‑03‑17)

1) `~/.codex/config.toml`
```
# Codex configuration for OpenAI (native)
model_reasoning_effort = "medium"

[shell_environment_policy]
inherit = "all"
```

2) `~/.codex/auth.json`
```
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "sk-...YOUR_OPENAI_KEY..."
}
```

3) Example Codex call
```
codex exec --sandbox danger-full-access --model gpt-5-mini "echo hello > hello.txt"
```

### B) MiniMax M2.5 via OpenRouter

1) `~/.codex/config.toml`
```
# Codex configuration for MiniMax via OpenRouter
model_provider = "openrouter"
model_reasoning_effort = "medium"

[shell_environment_policy]
inherit = "all"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
requires_openai_auth = true
```

2) `~/.codex/auth.json`
```
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "sk-or-v1-...YOUR_OPENROUTER_KEY..."
}
```

3) Example Codex call
```
codex exec --sandbox danger-full-access --model minimax/minimax-m2.5 "printf 'OK'"
```

Notes:
- OpenRouter is OpenAI‑compatible, so Codex sees `OPENAI_API_KEY` and an OpenAI‑style base URL and Just Works.
- For OpenRouter’s OpenAI models, use `openai/...` slugs (e.g., `openai/gpt-5-mini`). For MiniMax, use `minimax/minimax-m2.5`.

## How This Repo Automates It

This repo’s runner (`run_codex_direct.sh`) does four things for you:

1) Selects provider and exports API env vars based on `--model`.
- OpenAI direct: exports `OPENAI_API_KEY` only.
- OpenRouter (MiniMax): exports `OPENAI_API_KEY` and `OPENAI_BASE_URL=https://openrouter.ai/api/v1`.

2) Writes the correct `~/.codex/config.toml` and `~/.codex/auth.json` for that provider.

3) Performs a preflight API call to confirm the model you requested is the model you get (aborts on mismatch).

4) Invokes Codex non‑interactively via `codex exec`, passing the model.

You can inspect the exact logic in `run_codex_direct.sh`.

## Memory/No‑Memory and Data Direction

The wrappers add memory context outside of Codex (in Python) and pass a single focused task to Codex. When you choose a non‑baseline ablation, the wrapper selects the memory directory based on the direction:
- backward → `data/back_trs`
- forward → `data/fwr_trs`

Baseline ablation injects no memory.

## System Architecture in This Repo (How It Works)

High‑level flow when you run `./run_codex_direct.sh`:

1) Argument parsing and selection
   - You pass: issue IDs (or a repo filter), an ablation, a direction, and a model.
   - The script can auto‑expand a repo slug (e.g., `org/repo`) to all matching issue IDs from `data/eval_set.jsonl`.

2) Provider bootstrap
   - Based on the model value, the runner decides whether the provider is OpenAI (GPT models) or OpenRouter (MiniMax and any `provider/model` slug).
   - It then writes `~/.codex/config.toml` and `~/.codex/auth.json` accordingly (OpenAI vs OpenRouter) and exports the appropriate env vars.

3) Preflight verification
   - A small Python snippet makes a 1‑shot API call and checks `response.model` against your requested model. If they don’t match, the run aborts (prevents silent routing to a different model).

4) Context building (Python)
   - The Python runner (`codex/scripts/run_codex_ci_repair.py`) loads cached CI failure context or regenerates it if missing.
   - If ablation ≠ baseline, it loads memory from `data/back_trs` or `data/fwr_trs` depending on direction and formats memory instructions.

5) Codex execution
   - For each problem, it calls `codex exec --sandbox danger-full-access --model <MODEL>` with the prompt (CI failure + optional memory) and captures Codex’s edits.

6) Results collection
   - Files land in `results/codex/<ablation>_<model>/<issue_id>/` with the diff (`patch.diff`), transcripts, and analysis JSON.

This design makes Codex stateless per run (provider config is written fresh each time) and keeps model/memory selection outside of Codex.

## One‑Time Setup for This Project

- Install Codex CLI: `npm install -g @openai/codex-cli`
- Create Python env: `python3 -m venv .venv-codex && source .venv-codex/bin/activate`
- Install deps: `pip install -r requirements-codex.txt -r requirements-shared.txt litellm python-dotenv`
- Put keys in `.env` at repo root:
  - `OPENAI_API_KEY=...` (for GPT models)
  - `OPENROUTER_API_KEY=...` (for MiniMax via OpenRouter)
- Ensure datasets/memory exist:
  - IDs: `data/eval_issue_ids.json`
  - Full dataset: `data/eval_set.jsonl`
  - Memory (backward): `data/back_trs`
  - Memory (forward): `data/fwr_trs`

## Running Experiments (This Repo’s Wrapper)

Syntax:
```
./run_codex_direct.sh "<issue_ids|''>" <ablation> <direction> <model> [repo_slug] [dataset]
```

Examples:
- All issues, all ablations, both directions (GPT‑5‑mini):
  - `./run_codex_direct.sh "" all both gpt-5-mini`
- MiniMax, full memory, backward:
  - `./run_codex_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5`
- Snapshot (GPT‑5.4‑mini), baseline on a specific repo from dataset:
  - `./run_codex_direct.sh "" baseline backward gpt-5.4-mini-2026-03-17 org/repo data/eval_set.jsonl`

Notes:
- Forward vs backward only changes which memory directory is loaded.
- The model determines provider config automatically.

## Adding Another Non‑OpenAI Model

Recommended path: use OpenRouter slugs and keep Codex in OpenRouter mode.

1) Confirm the model slug on OpenRouter (e.g., `cohere/command-r-plus`, `deepseek/deepseek-chat`).
2) Run with that slug:
```
./run_codex_direct.sh "" baseline backward <provider>/<model_slug>
```
3) The runner will write `~/.codex` for OpenRouter and preflight the model. If the provider returns a different model name, the run aborts.

If you must call a provider’s own OpenAI‑compatible endpoint directly (without OpenRouter), adapt the runner’s case block to export `OPENAI_API_KEY` and `OPENAI_BASE_URL` for that provider, and write a matching `~/.codex/config.toml` (no `model_provider` needed if the endpoint is fully OpenAI‑compatible). For maintainability, we standardize on OpenRouter.

## Results Layout

```
results/codex/
  <ablation>_<model>/
    <issue_id>/
      checkout/
      ci_failure.json
      ci_failure.md
      ci_verification.json
      memory_context.md     # for memory runs
      issue_document_problem_*.md
      codex_transcript_problem_*.txt
      patch.diff
      result.json
```

## Troubleshooting & Diagnostics

- Preflight mismatch
  - The runner prints the requested vs returned `response.model` and aborts if they differ. Check your model slug and provider config.

- “Model metadata … fallback metadata”
  - From the client library (e.g., LiteLLM) only; the call still hits your requested model. Upgrade the client to remove the warning.

- Auth issues (401/403)
  - Verify `~/.codex/auth.json` and that your `.env` keys are loaded; ensure nothing is overriding `OPENAI_BASE_URL` incorrectly.

- Verify provider at runtime
  - Add `echo` lines or inspect runner logs to see which `~/.codex` files were written and which `OPENAI_BASE_URL` is in effect.

## Security

- Do not commit secrets. This repo ignores `codex/.codex-config/**`.
- Prefer `~/.codex` for provider config; it keeps keys out of the repo and lets multiple projects reuse the same setup.

## Security and Project‑Local Config (optional)

- Do not commit real keys. If you insist on project‑local config under `codex/.codex-config/`, keep `auth.json` out of Git (this repo’s `.gitignore` already ignores `codex/.codex-config/**`).
- We recommend using `~/.codex` so multiple projects can reuse the same credentials and you avoid secret files in the repo.

## Troubleshooting

- “Model metadata not found … defaulting to fallback metadata”
  - This comes from a client library’s local model table (e.g., LiteLLM), not Codex. The call still uses `minimax/minimax-m2.5`; only local heuristics (token budgets, pricing readouts) may be conservative. Upgrading the client usually removes the warning.

- Preflight fails (“Model mismatch”)
  - The provider returned a different `response.model` than requested. Check the model slug (e.g., ensure `minimax/minimax-m2.5`) and that the provider is set correctly in `~/.codex/config.toml`.

- 401/403 authentication
  - Verify `~/.codex/auth.json` has the right key and that `OPENAI_API_KEY`/`OPENAI_BASE_URL` env vars are not overriding them incorrectly.

## Quick Reference

- OpenAI GPT‑5 models
  - Provider: OpenAI native
  - Model: `gpt-5-mini`, `gpt-5.4-mini-2026-03-17`
  - Auth: `OPENAI_API_KEY=sk-...`

- MiniMax M2.5 (via OpenRouter)
  - Provider: OpenRouter
  - Model: `minimax/minimax-m2.5`
  - Auth: `OPENAI_API_KEY=sk-or-v1-...` and `base_url=https://openrouter.ai/api/v1`

- Invocation pattern
  - `codex exec --sandbox danger-full-access --model <MODEL> "<task or quoted instruction>"`
