# CI‑Repair‑Bench – Clean Setup & Run (Mini‑SWE + Codex)

This repo supports two agents:
- `mini-swe-agent`
- `codex` (OpenAI Codex CLI–based)

Agent models (exactly these three):
- `gpt-5-mini` (OpenAI)
- `gpt-5.4-mini-2026-03-17` (OpenAI snapshot)
- `minimax2.5` (MiniMax M2.5 via OpenRouter; we map to `minimax/minimax-m2.5`)

GLM 5.2 is allowed only for decomposition + memory build (not for agents).

Ablations: `BASELINE`, `L1`, `L1+L2`, `L1+L2+L3`
Directions: `backward` (data/back_trs) or `forward` (data/fwr_trs)

## 0) One‑Time Installation

- Root tools env
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-shared.txt
deactivate
```

- Mini‑SWE‑Agent env
```bash
cd miniswe-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
deactivate
cd ..
```

- Codex CLI + orchestration env
```bash
# Codex CLI
npm install -g @openai/codex-cli
codex --version

# Orchestration env
python3 -m venv .venv-codex
source .venv-codex/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements-codex.txt -r requirements-shared.txt python-dotenv litellm
deactivate
```

- Keys in `.env`
```ini
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Optional for GLM decomposition only
GLM_API_KEY=...
```

## 1) Split dataset into Memory/Eval

```bash
python3 scripts/split_before_decomposition.py
```
Creates:
- data/memory_set.jsonl
- data/eval_set.jsonl
- data/memory_issue_ids.json
- data/eval_issue_ids.json (if missing, build it below)

If `data/eval_issue_ids.json` is missing:
```bash
python3 - <<PY2
import json
out=[]
with open(data/eval_set.jsonl,r,encoding=utf-8) as f:
  for line in f:
    line=line.strip()
    if line:
      out.append(json.loads(line).get(id))
with open(data/eval_issue_ids.json,w,encoding=utf-8) as f:
  json.dump(out, f, indent=2)
print(wrote
