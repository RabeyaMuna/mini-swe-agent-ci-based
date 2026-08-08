# Codex CI Repair – Multi‑Model, Memory/No‑Memory

This project runs OpenAI Codex CLI as a repair agent over a CI‑repair benchmark, with or without memory, across multiple models. The runner configures Codex automatically per model and supports ablation=all and direction=both.

**Supported models**
- `gpt-5-mini` (OpenAI)
- `gpt-5.4-mini-2026-03-17` (OpenAI snapshot)
- `minimax/minimax-m2.5` (MiniMax via OpenRouter)

**Ablations**: `baseline`, `L1`, `L2`, `L3`, `L1+L2`, `L1+L2+L3`, or `all`

**Directions**: `backward`, `forward`, or `both`

Results are written to `results/codex/<ablation>_<model>/...` with model included in the folder name.

## Prerequisites

- Codex CLI: `npm install -g @openai/codex-cli`
- Python env for the runner and utilities:
  ```bash
  python3 -m venv .venv-codex
  source .venv-codex/bin/activate
  pip install -r requirements-codex.txt -r requirements-shared.txt litellm python-dotenv
  # Mini‑SWE‑Agent package (used by run_miniswe_direct.sh)
  pip install -e miniswe-agent
  ```
- API keys in `.env` (at repo root):
  ```ini
  # For OpenAI models
  OPENAI_API_KEY=sk-...
  # For MiniMax via OpenRouter
  OPENROUTER_API_KEY=sk-or-v1-...
  OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
  ```

## One‑Line “Everything” Run

Runs ALL ablations in BOTH directions for ALL issues found in `data/eval_issue_ids.json` using GPT‑5‑mini by default:

```bash
./run_codex_direct.sh
```

To target a different model, ablation, or direction:

```bash
./run_codex_direct.sh "" all both minimax/minimax-m2.5
./run_codex_direct.sh "" L1+L2+L3 backward gpt-5.4-mini-2026-03-17
```

## Codex: Full Command Reference (All Ablations × Directions)

Run against ALL issues in `data/eval_issue_ids.json` by passing an empty first arg `""`.

### GPT‑5‑mini (OpenAI)

Backward (uses `data/back_trs`):

```bash
./run_codex_direct.sh "" baseline   backward gpt-5-mini
./run_codex_direct.sh "" L1         backward gpt-5-mini
./run_codex_direct.sh "" L2         backward gpt-5-mini
./run_codex_direct.sh "" L3         backward gpt-5-mini
./run_codex_direct.sh "" L1+L2      backward gpt-5-mini
./run_codex_direct.sh "" L1+L2+L3   backward gpt-5-mini
```

Forward (uses `data/fwr_trs`):

```bash
./run_codex_direct.sh "" baseline   forward gpt-5-mini
./run_codex_direct.sh "" L1         forward gpt-5-mini
./run_codex_direct.sh "" L2         forward gpt-5-mini
./run_codex_direct.sh "" L3         forward gpt-5-mini
./run_codex_direct.sh "" L1+L2      forward gpt-5-mini
./run_codex_direct.sh "" L1+L2+L3   forward gpt-5-mini
```

### GPT‑5.4‑mini‑2026‑03‑17 (OpenAI snapshot)

Backward:

```bash
./run_codex_direct.sh "" baseline   backward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L1         backward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L2         backward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L3         backward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L1+L2      backward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L1+L2+L3   backward gpt-5.4-mini-2026-03-17
```

Forward:

```bash
./run_codex_direct.sh "" baseline   forward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L1         forward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L2         forward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L3         forward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L1+L2      forward gpt-5.4-mini-2026-03-17
./run_codex_direct.sh "" L1+L2+L3   forward gpt-5.4-mini-2026-03-17
```

### MiniMax M2.5 (via OpenRouter)

Backward:

```bash
./run_codex_direct.sh "" baseline   backward minimax/minimax-m2.5
./run_codex_direct.sh "" L1         backward minimax/minimax-m2.5
./run_codex_direct.sh "" L2         backward minimax/minimax-m2.5
./run_codex_direct.sh "" L3         backward minimax/minimax-m2.5
./run_codex_direct.sh "" L1+L2      backward minimax/minimax-m2.5
./run_codex_direct.sh "" L1+L2+L3   backward minimax/minimax-m2.5
```

Forward:

```bash
./run_codex_direct.sh "" baseline   forward minimax/minimax-m2.5
./run_codex_direct.sh "" L1         forward minimax/minimax-m2.5
./run_codex_direct.sh "" L2         forward minimax/minimax-m2.5
./run_codex_direct.sh "" L3         forward minimax/minimax-m2.5
./run_codex_direct.sh "" L1+L2      forward minimax/minimax-m2.5
./run_codex_direct.sh "" L1+L2+L3   forward minimax/minimax-m2.5
```

### Run Mini‑SWE‑Agent

Use the symmetric runner for Mini‑SWE. It accepts the same arguments and will select memory roots based on `direction`.

```bash
# All ablations + both directions on all issues (default model gpt‑5‑mini)
./run_miniswe_direct.sh                    # default workers=1
./run_miniswe_direct.sh "" all both gpt-5-mini '' '' 4   # 4 workers

# MiniMax, full memory, backward on all issues
./run_miniswe_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5

# GPT‑5.4‑mini snapshot, baseline only on a specific repo
./run_miniswe_direct.sh "" baseline backward gpt-5.4-mini-2026-03-17 octo-org/demo-repo data/eval_set.jsonl
```

Mini‑SWE using your existing `run_eval.py` workflow:

```bash
# Baseline (no memory), selected repos (MiniMax)
python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,camel \
    --ablation BASELINE \
    --model minimax \
    --direction backward \
    --workers 4

python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,flower,camel \
    --ablation BASELINE \
    --model minimax \
    --direction backward \
    --workers 4

# Full memory (L1+L2+L3), backward
python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,camel \
    --ablation L1+L2+L3 \
    --model minimax \
    --direction backward \
    --workers 4

# Partial memory ablations, backward
python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,flower,camel \
    --ablation L1+L2 \
    --model minimax \
    --direction backward \
    --workers 4

python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,flower,camel \
    --ablation L1 \
    --model minimax \
    --direction backward \
    --workers 4

# Forward memory runs: switch --direction forward (uses data/fwr_trs)
python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,camel \
    --ablation L1+L2+L3 \
    --model minimax \
    --direction forward \
    --workers 4

# Using GPT models with Mini‑SWE
python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,camel \
    --ablation BASELINE \
    --model gpt-5-mini \
    --direction backward \
    --workers 4

python3 scripts/run_eval.py \
    --issue-ids-file data/eval_issue_ids.json \
    --repos crewai,camel \
    --ablation L1+L2+L3 \
    --model gpt-5.4-mini-2026-03-17 \
    --direction backward \
    --workers 4
```

### Run Both Agents (Codex + Mini‑SWE)

Run both agents with one command:

```bash
./run_both_agents.sh "" all both gpt-5-mini           # Codex:1, Mini‑SWE:1
# Or with MiniMax across both agents
./run_both_agents.sh "" all both minimax/minimax-m2.5
```

## Selecting Issues

The runner accepts explicit IDs, a repo filter, or defaults to `data/eval_issue_ids.json`.

- Explicit IDs (comma‑separated):
  ```bash
  ./run_codex_direct.sh 129,130 baseline backward gpt-5-mini
  ```

- All IDs from `data/eval_issue_ids.json` (leave first arg empty):
  ```bash
  ./run_codex_direct.sh "" all both gpt-5-mini
  ```

- Filter by repository slug from dataset (auto‑expands to matching IDs):
  ```bash
  ./run_codex_direct.sh "" all both minimax/minimax-m2.5 octo-org/demo-repo data/eval_set.jsonl
  ```
  The same repo filter works for Mini‑SWE with `run_miniswe_direct.sh` and for both with `run_both_agents.sh`.

Dataset default is `data/eval_set.jsonl`. If missing, the Python runner can pull from Hugging Face when configured.

## Memory Configuration

- Ablation = `baseline` → no memory injection
- Ablation ≠ `baseline` → memory is injected automatically
  - `backward` uses `data/back_trs`
  - `forward` uses `data/fwr_trs`

You can run all memory levels and both directions by using `all` and `both`.

## What the Runner Does

- Auto‑writes Codex provider config to `~/.codex` per model (OpenAI direct for GPT models; OpenRouter for MiniMax).
- Validates the requested model by hitting the API once (pre‑flight).
- Builds memory/no‑memory prompts and invokes Codex via `codex exec`.
- Organizes outputs under `results/codex/` with model suffixes.

## Examples

- Single issue, full memory, backward (GPT‑5.4‑mini snapshot):
  ```bash
  ./run_codex_direct.sh 129 L1+L2+L3 backward gpt-5.4-mini-2026-03-17
  ```

- MiniMax M2.5 across all issues, both directions, all ablations:
  ```bash
  ./run_codex_direct.sh "" all both minimax/minimax-m2.5
  ```

- GPT‑5‑mini, forward memory only across all issues:
  ```bash
  ./run_codex_direct.sh "" L1+L2+L3 forward gpt-5-mini
  ```

## Outputs

Each run writes a structured folder per ablation and model with:
- `checkout/` – repository at failing commit
- `ci_failure.json`, `ci_failure.md`
- `ci_verification.json`
- `memory_context.md` (for memory runs)
- `issue_document_problem_*.md` (prompts)
- `codex_transcript_problem_*.txt`
- `patch.diff`
- `result.json`

## Notes

- Local Codex state and keys live in `~/.codex`; repo‑local `codex/.codex-config/**` is ignored by git.
- Ensure `.env` has the right key for the model you run: `OPENAI_API_KEY` for GPT‑5, `OPENROUTER_API_KEY` for MiniMax.
- To smoke‑test prompt generation without calling Codex: add `--dry-run` to the Python command in `run_codex_direct.sh` (or run the script once with no network to see pre‑flight fail early).
