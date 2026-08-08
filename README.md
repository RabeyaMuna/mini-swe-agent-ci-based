# CI‑Repair Bench – Clean Setup & Run (Mini‑SWE + Codex)

This project runs two agents over a CI‑repair benchmark, with or without memory, across three models. One agent at a time.
- mini‑swe‑agent
- codex (OpenAI Codex CLI)

Supported agent models (exact list):
- gpt-5-mini (OpenAI)
- gpt-5.4-mini-2026-03-17 (OpenAI snapshot)
- minimax/minimax-m2.5 (MiniMax M2.5 via OpenRouter)

Ablations: BASELINE, L1, L1+L2, L1+L2+L3
Directions: backward (data/back_trs) or forward (data/fwr_trs)

Notes
- GLM 5.2 may be used only for decomposition/memory build, not as an agent model.
- predictions.json is updated incrementally after each issue so crashes do not lose patches.

## 0) Install Once

- Create env and install shared deps
```bash
python3 -m venv .venv-codex
source .venv-codex/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements-shared.txt -r requirements-codex.txt python-dotenv litellm
# Install Mini‑SWE‑Agent (for the Mini‑SWE runner)
pip install -e miniswe-agent
```

- Install Codex CLI
```bash
npm install -g @openai/codex-cli
codex --version
```

- Put keys in .env at repo root
```ini
# OpenAI (for GPT‑5 models)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# OpenRouter (for MiniMax M2.5)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional only for decomposition/memory build
GLM_API_KEY=...
```

## 1) Prepare Data (Split → Decompose → Build Memory)

- Split dataset into memory/eval
```bash
python3 scripts/split_before_decomposition.py
```
Creates data/memory_set.jsonl and data/eval_set.jsonl (plus id lists).

- Backward decomposition (for backward memory builds)
```bash
python3 scripts/decompose_backward.py \
  --dataset data/memory_set.jsonl \
  --output  data/backward_decomposed.json
```

- Forward decomposition (optional, for forward memory)
```bash
python3 commit_decomposition/run_commit_decomposition.py \
  --dataset data/memory_set.jsonl \
  --output  data/commit_decomposed.json
```

- Build memory artifacts
```bash
# Backward memory (used when direction=backward)
python3 scripts/build_memory_l1_l2_l3.py --direction backward

# Forward memory (used when direction=forward)
python3 scripts/build_memory_l1_l2_l3.py --direction forward
```

## 2) Run Mini‑SWE‑Agent (evaluation runner)

Syntax
```bash
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos <comma repos> \
  --ablation <BASELINE|L1|L1+L2|L1+L2+L3> \
  --model <minimax2.5|gpt-5-mini|gpt-5.4-mini-2026-03-17> \
  --direction <backward|forward> \
  --workers <N>
```

Examples
```bash
# Baseline (no memory), backward, MiniMax
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,camel \
  --ablation BASELINE \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# Baseline, backward (3 repos)
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,flower,camel \
  --ablation BASELINE \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# Full memory L1+L2+L3, backward
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,camel \
  --ablation L1+L2+L3 \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# L1+L2, backward (3 repos)
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,flower,camel \
  --ablation L1+L2 \
  --model minimax2.5 \
  --direction backward \
  --workers 4

# L1 only, backward (3 repos)
python3 scripts/run_eval.py \
  --issue-ids-file data/eval_issue_ids.json \
  --repos crewai,flower,camel \
  --ablation L1 \
  --model minimax2.5 \
  --direction backward \
  --workers 4
```
Notes
- Use one agent at a time (Mini‑SWE or Codex, not both concurrently).
- --workers controls parallel issues per run (default 1). Try 4 or 6 on bigger machines.
- Valid Mini‑SWE models: minimax2.5, gpt‑5‑mini, gpt‑5.4‑mini‑2026‑03‑17.

## 3) Run Codex (OpenAI Codex CLI agent)

Key points
- The wrapper auto‑configures the provider per model and writes ~/.codex on the fly.
- Auth banner prints: auth mode, provider, endpoint, and CODEX_HOME.
- Supported models: gpt‑5‑mini, gpt‑5.4‑mini‑2026‑03‑17, minimax/minimax‑m2.5.
- Aliases: passing minimax or minimax2.5 is routed to minimax/minimax‑m2.5 automatically.

Syntax
```bash
./run_codex_direct.sh "<ids or empty>" <ablation> <direction> <model> [repo_slug] [dataset] [workers]
```

Examples
```bash
# All issues, ALL ablations, BOTH directions, GPT‑5‑mini, 4 workers
./run_codex_direct.sh "" all both gpt-5-mini    data/eval_set.jsonl  4

# MiniMax M2.5 via OpenRouter, full memory, backward
./run_codex_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5  data/eval_set.jsonl 4

# Snapshot model, baseline, backward, 6 workers
./run_codex_direct.sh "" baseline backward gpt-5.4-mini-2026-03-17  data/eval_set.jsonl 6

# Filter issues by repo slug (use dataset to expand IDs)
./run_codex_direct.sh "" L1 backward minimax/minimax-m2.5 agno-agi/agno data/eval_set.jsonl 4
```

Outputs
- results/codex/<ablation>_<model>/issue_id/ contains checkout, transcripts, patch.diff, result.json
- predictions.json is updated after each issue to avoid data loss on crashes
- For multi‑problem issues, Codex fixes one problem at a time in the same checkout and writes one unified patch.diff

## 4) Non‑OpenAI with Codex (MiniMax M2.5)

See codex/docs/reademe.md for how MiniMax M2.5 is wired via OpenRouter (OpenAI‑compatible) into Codex.

## 5) Troubleshooting

- GPT‑5 “max_tokens not supported” → Fixed: preflight uses Responses (max_output_tokens / max_completion_tokens) or Chat as needed.
- “Model metadata … fallback metadata” → Local client warning only; calls still hit minimax/minimax‑m2.5.
- “Model mismatch” FATAL → Provider returned a different model; correct the model string or provider, then rerun.
- ChatGPT login is not required; runs use API keys from .env.


### Mini‑SWE – 12 One‑Liners

Run ALL eval_issues for each model, ablation, and direction (workers=4, dataset=data/eval_set.jsonl).

# GPT‑5‑mini
./run_miniswe_direct.sh "" BASELINE backward gpt-5-mini "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" BASELINE forward  gpt-5-mini "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" L1+L2+L3 backward gpt-5-mini "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" L1+L2+L3 forward  gpt-5-mini "" data/eval_set.jsonl 4

# GPT‑5.4‑mini‑2026‑03‑17
./run_miniswe_direct.sh "" BASELINE backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" BASELINE forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" L1+L2+L3 backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4

# MiniMax M2.5
./run_miniswe_direct.sh "" BASELINE backward minimax2.5 "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" BASELINE forward  minimax2.5 "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" L1+L2+L3 backward minimax2.5 "" data/eval_set.jsonl 4
./run_miniswe_direct.sh "" L1+L2+L3 forward  minimax2.5 "" data/eval_set.jsonl 4

### Codex – 12 One‑Liners

Run ALL eval_issues for each model, ablation, and direction (workers=4, dataset=data/eval_set.jsonl). The script auto‑configures provider and prints an auth banner.

# GPT ‑5 ‑mini
./run_codex_direct.sh "" baseline backward gpt-5-mini "" data/eval_set.jsonl 4
./run_codex_direct.sh "" baseline forward  gpt-5-mini "" data/eval_set.jsonl 4
./run_codex_direct.sh "" L1+L2+L3 backward gpt-5-mini "" data/eval_set.jsonl 4
./run_codex_direct.sh "" L1+L2+L3 forward  gpt-5-mini "" data/eval_set.jsonl 4

# GPT ‑5.4 ‑mini ‑2026 ‑03 ‑17
./run_codex_direct.sh "" baseline backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
./run_codex_direct.sh "" baseline forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
./run_codex_direct.sh "" L1+L2+L3 backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4
./run_codex_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 4

# MiniMax M2.5 (via OpenRouter)
./run_codex_direct.sh "" baseline backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
./run_codex_direct.sh "" baseline forward  minimax/minimax-m2.5 "" data/eval_set.jsonl 4
./run_codex_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
./run_codex_direct.sh "" L1+L2+L3 forward  minimax/minimax-m2.5 "" data/eval_set.jsonl 4

