# CI‑Repair Bench – Clean Setup & Run (Mini‑SWE + Codex)

This project runs two agents over a CI‑repair benchmark, with or without memory, across the supported models. One agent at a time.
- mini‑swe‑agent
- codex (OpenAI Codex CLI)

Common agent models:
- **gpt-5.4-mini** (routed directly to OpenAI)
- `minimax2.5` (routed through OpenRouter; canonical Codex slug `minimax/minimax-m2.5`)
- **deepseek-v4-flash** (routed through OpenRouter; 1M context, 384K output - most capable model)

Ablations: BASELINE, L1, L1+L2, L1+L2+L3

Memory directions: backward (`data/back_trs`), forward (`data/fwr_trs`), or
bidirectional (`data/bidirect_trs`). Direction never applies to BASELINE;
baseline uses `none` because it does not retrieve memory.

Notes
- GLM 5.2 may be used only for decomposition/memory build, not as an agent model.
- predictions.json is updated incrementally after each issue so crashes do not lose patches.

## Recommended conflict-free setup

From any directory, run the project installer by absolute or relative path:

```bash
bash /home/ubuntu/Documents/mini-swe-agent-ci-based/INSTALL.sh
cd /home/ubuntu/Documents/mini-swe-agent-ci-based
source .venv-codex/bin/activate
python scripts/verify_installation.py
```

`INSTALL.sh` creates a clean `.venv-codex`, installs the pinned Python 3.13
dependency set from `requirements-codex.txt`, preserves the verified CPU
PyTorch pair, installs Mini-SWE in editable mode, and runs both import/version
checks and `pip check`. Activation cannot persist after a Bash script exits,
so run the displayed `source` command before using bare `python` or `python3`.

## 0A) Mini-SWE-Agent-only Setup (Codex Not Required)

Use this setup when you want to run only Mini-SWE-Agent. It does **not**
install or require the Codex CLI, Node.js, a ChatGPT login, or Docker.

### Prerequisites

- Python 3.10 or newer
- Git
- Internet access for model API calls and cloning benchmark repositories

On Ubuntu, install the system prerequisites with:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

### Create a standalone Mini-SWE environment

Run these commands from the repository root:

```bash
python3 -m venv .venv-miniswe
source .venv-miniswe/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e ./miniswe-agent
```

The editable Mini-SWE-Agent install includes the runner dependencies such as
LiteLLM, `datasets`, the memory backends, and embedding libraries. You do not
need `requirements-codex.txt` for Mini-SWE-Agent.

Activate this environment again whenever you open a new terminal:

```bash
source .venv-miniswe/bin/activate
```

### Configure the model API key

Create the local environment file:

```bash
cp -n .env.example .env
```

Then add the key for the model you will use:

```ini
# Required for --model gpt-5.4-mini
OPENAI_API_KEY=your-openai-api-key-here

# Required for --model minimax2.5
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Only one provider key is required when you run only one model. Never commit
the `.env` file.

### Verify the Mini-SWE installation

```bash
python -c "import minisweagent, litellm, datasets; print('Mini-SWE-Agent dependencies are installed')"
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py --help
test -f data/eval_set.jsonl && echo "Evaluation dataset found"
```

For a first API-backed smoke run, use one issue and one worker:

```bash
set -a
source .env
set +a

PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '^(43)$' \
  --ablation baseline \
  --direction none \
  --model gpt-5.4-mini \
  --workers 1
```

Replace `43` with an `instance_id` present in `data/eval_set.jsonl`. Baseline
runs do not require memory files. The L1, L1+L2, and L1+L2+L3 modes require
the appropriate files under `data/back_trs/` or `data/fwr_trs/`.

Mini-SWE-Agent clones each benchmark repository and runs its repair commands
locally. A particular benchmark project may therefore require its own tools
(for example `uv`, Poetry, Node.js, or Rust), even though those tools are not
required by the Mini-SWE runner itself. Repositories are cached under `repo/`;
set `MSWEA_REPO_CACHE_ROOT` to use a different cache directory.

## 0B) Combined Mini-SWE + Codex Setup

### macOS / Local Development Setup

- Create env and install shared deps
```bash
python3 -m venv .venv-codex
source .venv-codex/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements-codex.txt
# Install Mini‑SWE‑Agent (for the Mini‑SWE runner)
pip install -e miniswe-agent
```

- Install Codex CLI
```bash
npm install -g @openai/codex-cli
codex --version
```

- Create `.env` file at repo root with your API keys:
```ini
# OpenAI (for GPT-5.4 models)
OPENAI_API_KEY=your-openai-api-key-here

# OpenRouter (for MiniMax M2.5)
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# GLM-5.2 via Z.ai (optional, for decomposition/memory build)
GLM_API_KEY=your-glm-api-key-here
GLM_BASE_URL=https://api.z.ai/api/paas/v4
GLM_MODEL_NAME=glm-5.2

# HuggingFace (for dataset loading)
HUGGINGFACE_TOKEN=your-huggingface-token-here
```

> **Security**: Never commit `.env` file to git. It's already in `.gitignore`.

### Ubuntu Server / Remote Setup

**Prerequisites:**
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.10+
- Node.js 18+ (for Codex CLI)
- Git

**Step 1: Clone Repository**
```bash
# SSH into your server
ssh ubuntu@your-server-ip

# Clone the repo
cd ~/Documents
git clone https://github.com/your-username/mini-swe-agent-ci-based.git
cd mini-swe-agent-ci-based
```

**Step 2: Install System Dependencies**
```bash
# Update package list
sudo apt update

# Install Python 3.10+ and pip
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install Node.js 18+ (for Codex CLI)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Git if not present
sudo apt install -y git curl
```

**Step 3: Create Python Virtual Environment**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip wheel
pip install -r requirements-codex.txt

# Install Mini-SWE-Agent
pip install -e miniswe-agent
```

**Step 4: Install Codex CLI**
```bash
# Install globally
npm install -g @openai/codex-cli

# Verify installation
codex --version
```

**Step 5: Configure Environment Variables**
```bash
# Create .env file (copy API keys from secure location - DO NOT commit to git)
cat > .env << 'EOF'
# OpenAI API Key (for GPT-5.4)
OPENAI_API_KEY=your-openai-api-key-here

# OpenRouter API Key (for MiniMax M2.5)
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# GLM-5.2 via Z.ai Platform (optional)
GLM_API_KEY=your-glm-api-key-here
GLM_BASE_URL=https://api.z.ai/api/paas/v4
GLM_MODEL_NAME=glm-5.2

# HuggingFace Token
HUGGINGFACE_TOKEN=your-huggingface-token-here
EOF

# Secure the .env file
chmod 600 .env
```

> **Important**: Replace placeholder values with your actual API keys. Keep `.env` secure and never commit it.

**Step 6: Configure Codex (Ubuntu-specific)**
```bash
# Create Codex configuration directory
mkdir -p .codex

# Create config.toml for Codex
cat > .codex/config.toml << 'EOF'
model_reasoning_effort = "high"
approval_policy = "on-request"
sandbox_mode = "workspace-write"

[shell_environment_policy]
inherit = "all"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
requires_openai_auth = true
EOF
```

**Step 7: Verify Installation**
```bash
# Activate environment
source .venv/bin/activate

# Test Python dependencies
python3 -c "import litellm; print('✓ LiteLLM installed')"

# Test Codex CLI
codex --version

# Verify environment variables
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓ OPENAI_API_KEY:', 'set' if os.getenv('OPENAI_API_KEY') else 'missing')"
```

**Step 8: Running Jobs in Background (Recommended)**

Since server tasks can take hours, use `tmux` or `screen`:

```bash
# Install tmux
sudo apt install -y tmux

# Create a new session
tmux new -s decompose

# Run your command
source .venv/bin/activate
python3 scripts/decompose_backward.py --batch --dataset data/memory_set.jsonl --model minimax2.5 --output-dir data/back_trs

# Detach from session: Press Ctrl+B, then D
# Reattach later: tmux attach -t decompose
# List sessions: tmux ls
```

**Differences from macOS:**
- Use `bash` instead of `zsh` (though both work)
- Paths are `/home/ubuntu/...` instead of `/Users/...`
- Use `apt` for system packages instead of `brew`
- Configure firewall if running LiteLLM proxy: `sudo ufw allow 8000`
- For long-running tasks, use `tmux`, `screen`, or `nohup`

## 1) Prepare Data (Split → Decompose → Build Memory)

**Step 1: Split dataset into memory/eval**
```bash
source .venv-codex/bin/activate
python3 scripts/split_before_decomposition.py
```
Creates `data/memory_set.jsonl` and `data/eval_set.jsonl` (plus ID lists).

**Step 2: Decompose Issues (Choose One Approach)**

Each decomposition method automatically builds L1/L2/L3 memory. Choose based on your needs:

### **2a. Backward Decomposition (CI Failure → Problem)**
```bash
python3 scripts/decompose_backward.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model minimax2.5 \
  --output-dir data/back_trs
```
- Analyzes CI failures and traces backward to identify problems
- Saves to **`data/back_trs/`**

### **2b. Forward Decomposition (Commit → Problem)**
```bash
python3 scripts/decompose_commits.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model deepseek-v4-flash \
  --output-dir data/fwr_trs
```
- Analyzes commit changes and traces forward to identify problems
- Saves to **`data/fwr_trs/`**

### **2c. Bidirectional Decomposition (Unified View - Recommended)**
```bash
python3 scripts/decompose_bidirectional.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model gpt-4o-mini \
  --output-dir data/bidirect_trs
```
- Combines forward (commit-based) + backward (CI failure-based) analysis
- Reconciles both views using intelligent LLM-based synthesis
- Provides most comprehensive decomposition
- Saves to **`data/bidirect_trs/`**

**All methods automatically generate:**
- `decomposed_issues.json` (decomposed problems)
- `failure_memory.json` (L1 - failure sequences)
- `repo_memory.json` (L2 - repair strategies)
- `cross_memory.json` (L3 - universal patterns)

**Model Recommendations:**
- `minimax2.5` - Cost-effective, good for backward/simple analysis
- `deepseek-v4-flash` - Best for large-scale (1M context), forward decomposition
- `gpt-4o-mini` - Balanced quality/cost for bidirectional synthesis

**Additional Options (works with all three methods):**
```bash
# Process specific issues only
python3 scripts/decompose_bidirectional.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model gpt-4o-mini \
  --issue-ids "43,111,121" \
  --output-dir data/bidirect_trs

# Limit number of issues to process
python3 scripts/decompose_bidirectional.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model gpt-4o-mini \
  --limit 10 \
  --output-dir data/bidirect_trs

# Single issue mode (no --batch flag)
python3 scripts/decompose_bidirectional.py \
  --dataset data/memory_set.jsonl \
  --issue-ids "43" \
  --model gpt-4o-mini \
  --output-dir data/bidirect_trs
```

**Important Notes:**
- ✅ Memory building (L1/L2/L3) is **automatic** - no need to run separate scripts
- ✅ Remove `--use-huggingface` flag when using local `data/memory_set.jsonl`
- 📁 Output directories:
  - `data/back_trs/` = Backward (CI failure → problem)
  - `data/fwr_trs/` = Forward (commit → problem)  
  - `data/bidirect_trs/` = Bidirectional (forward + backward unified)

### Running Decomposition on Ubuntu Server

**SSH and Setup:**
```bash
ssh ubuntu@your-server-ip
cd /home/ubuntu/Documents/mini-swe-agent-ci-based
source .venv/bin/activate
```

**Three Decomposition Methods (using tmux for background execution):**

```bash
# 1. BACKWARD (minimax2.5) - CI failure → problem
tmux new -s backward
python3 scripts/decompose_backward.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model minimax2.5 \
  --output-dir data/back_trs
# Detach: Ctrl+B then D

# 2. FORWARD (deepseek-v4-flash) - Commit → problem
tmux new -s forward
python3 scripts/decompose_commits.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model deepseek-v4-flash \
  --output-dir data/fwr_trs
# Detach: Ctrl+B then D

# 3. BIDIRECTIONAL (gpt-4o-mini) - Unified forward + backward
tmux new -s bidirect
python3 scripts/decompose_bidirectional.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model gpt-4o-mini \
  --output-dir data/bidirect_trs
# Detach: Ctrl+B then D
```

**Or using nohup for detached execution:**
```bash
# Backward with minimax2.5
nohup python3 scripts/decompose_backward.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model minimax2.5 \
  --output-dir data/back_trs > backward.log 2>&1 &

# Forward with deepseek-v4-flash
nohup python3 scripts/decompose_commits.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model deepseek-v4-flash \
  --output-dir data/fwr_trs > forward.log 2>&1 &

# Bidirectional with gpt-4o-mini (limit 50 issues)
nohup python3 scripts/decompose_bidirectional.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model gpt-4o-mini \
  --limit 50 \
  --output-dir data/bidirect_trs > bidirect.log 2>&1 &

# Monitor progress
tail -f backward.log
tail -f forward.log
tail -f bidirect.log
```

**Supported Models:**
- `minimax2.5` - Cost-effective, good for backward/simple analysis
- `deepseek-v4-flash` - Best for large-scale (1M context), forward decomposition
- `gpt-4o-mini` - Balanced quality/cost for bidirectional synthesis
- `gpt-5.4-mini` - Premium option (requires OpenAI API)
- `glm5.2` - Alternative (requires GLM_API_KEY in .env)

Choose based on your needs:
- **Quick setup**: Use backward only
- **Comprehensive**: Use bidirectional (combines both views)
- **Experimental**: Compare all three approaches

## 2) Run Mini‑SWE‑Agent (evaluation runner)

Complete the [Mini-SWE-Agent-only setup](#0a-mini-swe-agent-only-setup-codex-not-required)
first, then activate its environment and load `.env`:

```bash
source .venv-miniswe/bin/activate
set -a
source .env
set +a
```

### Direct runner

This is the recommended command when Mini-SWE-Agent has its own environment:

```bash
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '<regular expression over instance_id>' \
  --ablation <baseline|L1|L1+L2|L1+L2+L3> \
  --direction <none|backward|forward|bidirectional> \
  --model <gpt-5.4-mini|minimax2.5> \
  --workers <N>
```

Examples:

```bash
# One issue, baseline (no memory), MiniMax through OpenRouter
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '^(43)$' \
  --ablation baseline \
  --model minimax2.5 \
  --direction none \
  --workers 1

# Several issues with full backward memory, GPT-5.4 Mini
PYTHONPATH=. python scripts/run_miniswe_ci_bench.py \
  --dataset data/eval_set.jsonl \
  --issue_regex '^(43|111|121)$' \
  --ablation L1+L2+L3 \
  --direction backward \
  --model gpt-5.4-mini \
  --workers 4
```

Results are written incrementally under:

```text
# For baseline (no directional memory):
results/miniswe-agent/baseline_<model>/

# For L1, L1+L2, L1+L2+L3 (with directional memory):
results/miniswe-agent/<direction>/<ablation>_<model>/
```

### Convenience wrapper

The wrapper accepts a comma-separated issue list and can run several ablations
or multiple memory directions:

```bash
./run_miniswe_direct.sh \
  "43,111,121" \
  L1+L2+L3 \
  backward \
  gpt-5.4-mini \
  "" \
  data/eval_set.jsonl \
  4
```

Arguments, in order, are:

```text
issue_ids  ablation  direction  model  optional_repo_slug  dataset  workers
```

Pass an empty issue list (`""`) to use every ID in
`data/eval_issue_ids.json`. Use direction `none` for baseline. For memory
ablations, use `backward`, `forward`, or `bidirectional`; `both` runs backward
and forward, while `all` runs all three memory directions. The optional fifth
argument also accepts comma-separated short repository names or exact
`owner/repo` slugs.

```bash
bash ./run_miniswe_direct.sh "" BASELINE none gpt-5.4-mini \
  "agno,axolotl,camel,crewai,django-import-export" data/eval_set.jsonl 4
```

> The wrapper automatically loads `.env` and prefers `.venv-miniswe`. It falls
> back to `.venv-codex` only when the dedicated Mini-SWE environment is absent.
> GPT models receive only the OpenAI credential; MiniMax and DeepSeek receive
> only the OpenRouter credential.

Verify Mini-SWE routing without making an API request:

```bash
MINISWE_CONFIG_ONLY=1 bash ./run_miniswe_direct.sh 1 baseline none gpt-5.4-mini
MINISWE_CONFIG_ONLY=1 bash ./run_miniswe_direct.sh 1 baseline none minimax2.5
```

Notes:

- Run Mini-SWE and Codex separately; concurrent jobs may contend for repository caches.
- `--workers` controls parallel issues. Start with `1`, then increase it based on CPU, memory, and API rate limits.
- Valid Mini-SWE model aliases include `gpt-5.4-mini`, `minimax2.5`, and `deepseek-v4-flash`.
- `baseline` has no direction, does not retrieve memory, and saves directly to `results/miniswe-agent/baseline_<model>/`.
- Other ablations (L1, L1+L2, L1+L2+L3) load backward, forward, or bidirectional memory according to `--direction` and save to `results/miniswe-agent/<direction>/<ablation>_<model>/`.

## 3) Run Codex (OpenAI Codex CLI agent)

### Key Points
- GPT/OpenAI and MiniMax/OpenRouter routing is automatic.
- The launcher loads `.env`; only the matching provider key is passed onward.
- Use `run_codex_direct.sh` for both providers.

### Direct Codex Commands (Provider Configuration Is Automatic)

After pulling the repository, put `OPENAI_API_KEY` and/or
`OPENROUTER_API_KEY` in the repository's `.env`. The launcher loads `.env`
itself. Do not manually edit `.codex-local/config.toml`.

Provider routing is selected from the model argument:

- `gpt-*`, `chatgpt-*`, `o<number>*`, and `codex-*` use OpenAI directly.
- `openai/<model>` is normalized to the native OpenAI model name.
- `minimax2.5`, `minimax-m2.5`, and `minimax/*` use OpenRouter.
- Each model gets an isolated `.codex-local/<model>/` configuration, so
  concurrent GPT and MiniMax runs cannot overwrite one another.
- Resume is enabled by default: issue IDs already present in that
  model/ablation's `predictions.json` are skipped. Set `CODEX_RESUME=0` only
  when you intentionally want to regenerate them.

You can verify routing without starting a benchmark or making an API request:

```bash
CODEX_CONFIG_ONLY=1 bash ./run_codex_direct.sh 1 baseline backward gpt-5.4-mini
CODEX_CONFIG_ONLY=1 bash ./run_codex_direct.sh 1 baseline backward minimax2.5
```

Then run Codex:
```bash
# Basic command with GPT-5.4-mini
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini "" data/eval_set.jsonl 1

# Full memory (L1+L2+L3)
bash ./run_codex_direct.sh "" L1+L2+L3 backward gpt-5.4-mini "" data/eval_set.jsonl 4

# MiniMax model
bash ./run_codex_direct.sh "" baseline backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
```

### Prompt caching and dynamic issues

The repair prompt is arranged as a stable instruction prefix followed by an
explicit dynamic-context boundary. Repository metadata, CI logs, problem
details, package/configuration evidence, verification data, and memory repair
plans remain in the dynamic suffix and are generated independently for every
issue.

OpenAI prompt caching is automatic for eligible requests. The runner prints a
line like this before each Codex problem:

```text
[Prompt cache] layout=stable_prefix_dynamic_suffix_v1 template=... stable_chars=... dynamic_chars=...
```

The template fingerprint covers only the stable prefix. Editing the repair
instructions automatically changes the fingerprint, so the edited prompt gets
a normal cache miss and becomes a new cacheable prefix. Dynamic issue changes
do not change that fingerprint and cannot reuse another issue's dynamic data.
When direct LiteLLM calls return cache accounting, their API log also includes
`cached_input=<tokens>`.

Caching does not cache answers and does not reduce the model's reasoning
effort. It discounts matching input tokens only; output and reasoning tokens
are still billed normally. A first request, an edited prefix, or a provider
without a matching recent prefix will run normally as a cache miss.

### Running on Server

```bash
# Using tmux (recommended for long-running jobs)
tmux new -s codex
source .venv-codex/bin/activate
./run_gpt54.sh L1+L2+L3 backward data/eval_set.jsonl 4
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t codex

# Or using nohup
nohup ./run_gpt54.sh baseline backward data/eval_set.jsonl 4 > codex.log 2>&1 &
tail -f codex.log
```

### View Results

```bash
# Results are in: results/codex/<direction>/<ablation>_<model>/<issue-id>/
ls -la results/codex/backward/baseline_gpt-5_4-mini/416/

# View specific result
cat results/codex/backward/baseline_gpt-5_4-mini/416/result.json | python3 -m json.tool

# View generated patch
cat results/codex/backward/baseline_gpt-5_4-mini/416/patch.diff
```

Syntax
```bash
bash ./run_codex_direct.sh "<ids or empty>" <ablation> <direction> <model> [repo_filters] [dataset] [workers]
```

Examples
```bash
# All issues, ALL ablations, BOTH directions, GPT‑5‑mini, 4 workers
bash ./run_codex_direct.sh "" all both gpt-5-mini "" data/eval_set.jsonl 4

# MiniMax M2.5 via OpenRouter, full memory, backward
bash ./run_codex_direct.sh "" L1+L2+L3 backward minimax2.5 "" data/eval_set.jsonl 4

# Snapshot model, baseline, backward, 6 workers
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini-2026-03-17 "" data/eval_set.jsonl 6

# Filter issues by repo slug (use dataset to expand IDs)
bash ./run_codex_direct.sh "" L1 backward minimax/minimax-m2.5 agno-agi/agno data/eval_set.jsonl 4

# Filter by several short repo names; short names include every matching owner
bash ./run_codex_direct.sh "" L1+L2+L3 forward minimax/minimax-m2.5 \
  "agno,axolotl,camel,crewai,django-import-export" data/eval_set.jsonl 4
```

Outputs
- results/codex/<direction>/<ablation>_<model>/issue_id/ contains checkout, transcripts, patch.diff, result.json
- predictions.json is updated after each issue to avoid data loss on crashes
- For multi‑problem issues, Codex fixes one problem at a time in the same checkout and writes one unified patch.diff

Direction-specific output directories keep backward and forward resume state
separate. Existing legacy results directly under `results/codex/` are preserved
but are not treated as completed by new direction-aware runs.

## 4) Non‑OpenAI with Codex (MiniMax M2.5)

See codex/docs/reademe.md for how MiniMax M2.5 is wired via OpenRouter (OpenAI‑compatible) into Codex.

## 5) Decomposition – Quick Commands

Three decomposition approaches with different model recommendations:

#### **Backward Decomposition** (CI failure → problem) - minimax2.5
```bash
# All issues
python3 scripts/decompose_backward.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model minimax2.5 \
  --output-dir data/back_trs

# Specific issues
python3 scripts/decompose_backward.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model minimax2.5 \
  --issue-ids “43,111,121” \
  --output-dir data/back_trs
```

#### **Forward Decomposition** (commit → problem) - deepseek-v4-flash
```bash
# All issues (1M context - handles large datasets)
python3 scripts/decompose_commits.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model deepseek-v4-flash \
  --output-dir data/fwr_trs

# Limited batch (50 issues)
python3 scripts/decompose_commits.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model deepseek-v4-flash \
  --limit 50 \
  --output-dir data/fwr_trs
```

#### **Bidirectional Decomposition** (unified forward + backward) - gpt-4o-mini
```bash
# All issues (recommended - most comprehensive)
python3 scripts/decompose_bidirectional.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model gpt-4o-mini \
  --output-dir data/bidirect_trs

# Specific issues
python3 scripts/decompose_bidirectional.py \
  --batch \
  --dataset data/memory_set.jsonl \
  --model gpt-4o-mini \
  --issue-ids “43,111,121,416” \
  --output-dir data/bidirect_trs

# Single issue (no --batch)
python3 scripts/decompose_bidirectional.py \
  --dataset data/memory_set.jsonl \
  --issue-ids “43” \
  --model gpt-4o-mini \
  --output-dir data/bidirect_trs
```

**Output Structure** (each method creates):
- `decomposed_issues.json` - Decomposed problems
- `failure_memory.json` - L1 memory (failure sequences)
- `repo_memory.json` - L2 memory (repair strategies)
- `cross_memory.json` - L3 memory (universal patterns)

**Output Directories:**
- `data/back_trs/` - Backward traces
- `data/fwr_trs/` - Forward traces
- `data/bidirect_trs/` - Bidirectional traces (unified)

## 6) Troubleshooting

- GPT‑5 “max_tokens not supported” → Fixed: preflight uses Responses (max_output_tokens / max_completion_tokens) or Chat as needed.
- “Model metadata … fallback metadata” → Local client warning only; calls still hit minimax/minimax‑m2.5.
- “Model mismatch” FATAL → Provider returned a different model; correct the model string or provider, then rerun.
- ChatGPT login is not required; runs use API keys from .env.


### Mini‑SWE – Quick Commands

Run all evaluation issues for each model and ablation (workers=4,
dataset=`data/eval_set.jsonl`). Baseline runs once without a direction; memory
ablations can run backward, forward, or bidirectional.
The wrapper loads `.env` and selects OpenAI or OpenRouter automatically.

**Note:** Baseline uses direction `none` and saves to
`results/miniswe-agent/baseline_<model>/`.

#### GPT-5.4-mini (Recommended)
```bash
# Baseline (no direction or memory retrieval)
bash ./run_miniswe_direct.sh "" BASELINE none gpt-5.4-mini "" data/eval_set.jsonl 4

# Memory modes (results saved to results/miniswe-agent/<direction>/l1_l2_l3_gpt-5.4-mini/)
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 bidirectional gpt-5.4-mini "" data/eval_set.jsonl 4
```

#### MiniMax M2.5
```bash
# Baseline (no direction or memory retrieval)
bash ./run_miniswe_direct.sh "" BASELINE none minimax2.5 "" data/eval_set.jsonl 4

# Memory modes (results saved to results/miniswe-agent/<direction>/l1_l2_l3_minimax-m2.5/)
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  minimax2.5 "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 bidirectional minimax2.5 "" data/eval_set.jsonl 4
```

#### DeepSeek-V4-Flash (1M Context, Best for Large-scale)
```bash
# Baseline (no direction or memory retrieval)
bash ./run_miniswe_direct.sh "" BASELINE none deepseek-v4-flash "" data/eval_set.jsonl 4

# Memory modes (results saved to results/miniswe-agent/<direction>/l1_l2_l3_deepseek-v4-flash/)
bash ./run_miniswe_direct.sh "" L1+L2+L3 backward deepseek-v4-flash "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 forward  deepseek-v4-flash "" data/eval_set.jsonl 4
bash ./run_miniswe_direct.sh "" L1+L2+L3 bidirectional deepseek-v4-flash "" data/eval_set.jsonl 4
```
### Codex – Quick Commands

The launcher automatically loads `.env` and creates the correct provider configuration.

#### Manual Commands with GPT-5.4-mini
```bash
bash ./run_codex_direct.sh "" baseline backward gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 backward gpt-5.4-mini "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 forward  gpt-5.4-mini "" data/eval_set.jsonl 4

bash ./run_codex_direct.sh \
  "" \
  L1+L2+L3 \
  forward \
  minimax/minimax-m2.5 \
  "agno,axolotl,camel,crewai,django-import-export" \
  data/eval_set.jsonl \
  4

```

#### MiniMax M2.5 (via OpenRouter)
```bash
bash ./run_codex_direct.sh "" baseline none minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" baseline forward  minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 backward minimax/minimax-m2.5 "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 bidirectional  minimax/minimax-m2.5 "" data/eval_set.jsonl 4
```

#### DeepSeek-V4-Flash (via OpenRouter - 1M context, 384K output)
```bash
# Baseline
bash ./run_codex_direct.sh "" baseline backward deepseek-v4-flash "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" baseline forward  deepseek-v4-flash "" data/eval_set.jsonl 4

# Full memory (L1+L2+L3) - leverages massive 1M context window
bash ./run_codex_direct.sh "" L1+L2+L3 backward deepseek-v4-flash "" data/eval_set.jsonl 4
bash ./run_codex_direct.sh "" L1+L2+L3 forward  deepseek-v4-flash "" data/eval_set.jsonl 4

# Specific repositories with full memory
bash ./run_codex_direct.sh "" L1+L2+L3 backward deepseek-v4-flash \
  "agno,axolotl,camel,crewai,django-import-export" data/eval_set.jsonl 4
```

## Cost and time reports

Both benchmark runners continuously save cost and timing data in each
model/direction/ablation output directory:

- `run_metrics.json` keeps the durable per-instance attempt ledger and API-call
  checkpoints.
- `cost_time_report.json` contains per-instance totals and the overall run total.
- `cost_time_report.csv` contains the same per-instance totals for analysis.

Interrupted and failed attempts are retained. When the same issue is resumed,
the new attempt is appended and its cost/time is added to earlier attempts;
only an instance with a completed attempt and usable patch is skipped. Codex
model-preflight usage is saved separately under the direction's
`_run_overhead_<model>/` directory so it is not silently assigned to one issue.
While an attempt is running, a heartbeat checkpoints its elapsed time every ten
seconds, so even a forced process termination loses at most the time since the
last heartbeat. Graceful interruptions record the exact elapsed time.

`total_wall_time_seconds` is the sum of instance-attempt runtimes, including
retries. With parallel workers it represents total consumed instance time, not
the shorter batch makespan. For Codex, `total_api_time_seconds` measures the
complete Codex turn because the CLI publishes aggregate token usage when the
turn ends; Mini-SWE records individual model-request time.

`cost_complete` is false when a provider returned token usage but no billed
cost and the installed LiteLLM pricing registry could not price that model.
This avoids reporting an unknown charge as `$0`.

For a model missing from LiteLLM's registry, copy
`model-pricing.example.json`, enter the provider's per-million-token rates,
and set `RUN_METRICS_PRICING_FILE` to that file before launching the run.

Create a combined report across output directories with:

```bash
python3 scripts/analyze_costs.py results/codex \
  --group-by direction \
  --output results/codex/overall_cost_time_report.json
```

## Viewing API Costs

The system tracks **actual API costs** for all LLM calls across:
- Memory plugin (L1/L2/L3 decomposition, retrieval)
- Agent runs (Mini-SWE, Codex)
- Reuse checks and all other LLM utilities

Costs are persisted in `costs.json` files that survive interruptions and support reruns.

### View costs for a single run:

```bash
python scripts/view_costs.py results/miniswe-agent/backward/l1+l2+l3_deepseek-v4-flash/costs.json
```

### View aggregated costs across all runs:

```bash
python scripts/view_costs.py results/miniswe-agent/
```

**Example output:**

```
════════════════════════════════════════════════════════════════════════════════════════════════════
  Aggregated Costs from results/miniswe-agent/
════════════════════════════════════════════════════════════════════════════════════════════════════

L1+L2+L3:
  Model                          Direction       Instances  Total Cost      Avg/Instance   
  ------------------------------ --------------- ---------- --------------- ---------------
  deepseek-v4-flash              backward        408        $12.345678      $0.030259
  deepseek-v4-flash              forward         408        $11.234567      $0.027536
  minimax-m2.5                   bidirectional   408        $15.678901      $0.038428

════════════════════════════════════════════════════════════════════════════════════════════════════
  GRAND TOTAL: $39.259146
════════════════════════════════════════════════════════════════════════════════════════════════════
```

### Cost tracking features:

- **Per-instance costs**: Each instance's cost is tracked individually in `costs.json`
- **Interrupt-safe**: Costs persist across interruptions and resumes
- **Rerun support**: When an instance is rerun, its cost is replaced with the new value
- **Comprehensive**: Captures ALL API calls (memory, agent, retrieval, utilities)
- **Per-configuration**: Costs organized by model/ablation/direction

The final summary at the end of each run displays:

```
[CIBench] ═══════════════════════════════════════════════════════════════
[CIBench] COST SUMMARY (Actual API Cost):
[CIBench]   Total Cost:       $12.345678
[CIBench]   Instances Billed: 408
[CIBench]   Avg Cost/Instance: $0.030259
[CIBench]   Model:      deepseek-v4-flash
[CIBench]   Ablation:   L1+L2+L3
[CIBench]   Direction:  backward
[CIBench]   Cost File:  results/miniswe-agent/backward/l1+l2+l3_deepseek-v4-flash/costs.json
[CIBench] ═══════════════════════════════════════════════════════════════
```
