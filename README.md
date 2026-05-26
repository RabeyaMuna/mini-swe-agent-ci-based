<div align="center">
<a href="https://mini-swe-agent.com/latest/"><img src="https://github.com/SWE-agent/mini-swe-agent/raw/main/docs/assets/mini-swe-agent-banner.svg" alt="mini-swe-agent banner" style="height: 7em"/></a>
</div>

# The minimal AI software engineering agent

📣 [Run mini-swe-agent on our new & extremely challenging benchmark, ProgramBench](https://mini-swe-agent.com/latest/usage/programbench/)<br/>
📣 [New tutorial on building minimal AI agents](https://minimal-agent.com/)<br/>
📣 [Gemini 3 Pro reaches 74% on SWE-bench verified with mini-swe-agent!](https://x.com/KLieret/status/1991164693839270372)<br/>
📣 [New blogpost: Randomly switching between GPT-5 and Sonnet 4 boosts performance](https://www.swebench.com/post-250820-mini-roulette.html)

[![Docs](https://img.shields.io/badge/Docs-green?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://mini-swe-agent.com/latest/)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/swe-bench/shared_invite/zt-36pj9bu5s-o3_yXPZbaH2wVnxnss1EkQ)
[![PyPI - Version](https://img.shields.io/pypi/v/mini-swe-agent?style=for-the-badge&logo=python&logoColor=white&labelColor=black&color=deeppink)](https://pypi.org/project/mini-swe-agent/)

> [!WARNING]
> This is **mini-swe-agent v2**. Read the [migration guide](https://mini-swe-agent.com/latest/advanced/v2_migration/). For the previous version, check out the [v1 branch](https://github.com/SWE-agent/mini-swe-agent/tree/v1).

In 2024, we built [SWE-bench](https://github.com/swe-bench/SWE-bench) & [SWE-agent](https://github.com/swe-agent/swe-agent) and helped kickstart the coding agent revolution.

We now ask: **What if our agent was 100x simpler, and still worked nearly as well?**

`mini` is

- **Widely adopted**: Used by Meta, NVIDIA, Essential AI, IBM, Nebius, Anyscale, Princeton University, Stanford University, and many more.
- **Minimal**: Just some 100 lines of python for the [agent class](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py) (and a bit more for the [environment](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/local.py),
[model](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_model.py), and [run script](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/hello_world.py)) — no fancy dependencies!
- **Performant:** Scores >74% on the [SWE-bench verified benchmark](https://www.swebench.com/); starts much faster than Claude Code
- **Deployable:** Supports **local environments**, **docker/podman**, **singularity/apptainer**, **bublewrap**, **contree**, and more
- **Compatible:** Supports all models via **litellm**, **openrouter**, **portkey**, and more. Support for `/completion` and `/response` endpoints, interleaved thinking etc.
- Built by the Princeton & Stanford team behind [SWE-bench](https://swebench.com), [SWE-agent](https://swe-agent.com), and more
- **Tested:** [![Codecov](https://img.shields.io/codecov/c/github/swe-agent/mini-swe-agent?style=flat-square)](https://codecov.io/gh/SWE-agent/mini-swe-agent)

<details>

<summary>More motivation (for research)</summary>

[SWE-agent](https://swe-agent.com/latest/) jump-started the development of AI agents in 2024. Back then, we placed a lot of emphasis on tools and special interfaces for the agent.
However, one year later, as LMs have become more capable, a lot of this is not needed at all to build a useful agent!
In fact, the `mini` agent

- **Does not have any tools other than bash** — it doesn't even need to use the tool-calling interface of the LMs.
  This means that you can run it with literally any model. When running in sandboxed environments you also don't need to take care
  of installing a single package — all it needs is bash.
- **Has a completely linear history** — every step of the agent just appends to the messages and that's it.
  So there's no difference between the trajectory and the messages that you pass on to the LM.
  Great for debugging & fine-tuning.
- **Executes actions with `subprocess.run`** — every action is completely independent (as opposed to keeping a stateful shell session running).
  This makes it trivial to execute the actions in sandboxes (literally just switch out `subprocess.run` with `docker exec`) and to
  scale up effortlessly. Seriously, this is [a big deal](https://mini-swe-agent.com/latest/faq/#why-no-shell-session), trust me.

This makes it perfect as a baseline system and for a system that puts the language model (rather than
the agent scaffold) in the middle of our attention.
You can see the result on the [SWE-bench (bash only)](https://www.swebench.com/) leaderboard, that evaluates the performance of different LMs with `mini`.

</details>

<details>
<summary>More motivation (as a tool)</summary>

Some agents are overfitted research artifacts. Others are UI-heavy frontend monsters.

The `mini` agent wants to be a hackable tool, not a black box.

- **Simple** enough to understand at a glance
- **Convenient** enough to use in daily workflows
- **Flexible** to extend

Unlike other agents (including our own [swe-agent](https://swe-agent.com/latest/)), it is radically simpler, because it:

- **Does not have any tools other than bash** — it doesn't even need to use the tool-calling interface of the LMs.
  Instead of implementing custom tools for every specific thing the agent might want to do, the focus is fully on the LM utilizing the shell to its full potential.
  Want it to do something specific like opening a PR?
  Just tell the LM to figure it out rather than spending time to implement it in the agent.
- **Executes actions with `subprocess.run`** — every action is completely independent (as opposed to keeping a stateful shell session running).
  This is [a big deal](https://mini-swe-agent.com/latest/faq/#why-no-shell-session) for the stability of the agent, trust me.
- **Has a completely linear history** — every step of the agent just appends to the messages that are passed to the LM in the next step and that's it.
  This is great for debugging and understanding what the LM is prompted with.

</details>

<details>
<summary>Should I use SWE-agent or mini-SWE-agent?</summary>

You should consider `mini-swe-agent` your default choice.
In particular, you should use `mini-swe-agent` if

- You want a quick command line tool that works locally
- You want an agent with a very simple control flow
- You want even faster, simpler & more stable sandboxing & benchmark evaluations
- You are doing FT or RL and don't want to overfit to a specific agent scaffold

You should use `swe-agent` if

- You want to experiment with different sets of tools, each with their own interface
- You want to experiment with different history processors

What you get with both

- Excellent performance on SWE-Bench
- A trajectory browser

</details>

<table>
<tr>
<td width="50%">
<a href="https://mini-swe-agent.com/latest/usage/mini/"><strong>CLI</strong></a> (<code>mini</code>)
</td>
<td>
<a href="https://mini-swe-agent.com/latest/usage/swebench/"><strong>Batch inference</strong></a>
</td>
</tr>
<tr>
<td width="50%">

![mini](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini.gif?raw=true)

</td>
<td>

![swebench](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/swebench.gif?raw=true)

</td>
</tr>
<tr>
<td>
<a href="https://mini-swe-agent.com/latest/usage/inspector/"><strong>Trajectory browser</strong></a>
</td>
<td>
<a href="https://mini-swe-agent.com/latest/advanced/cookbook/"><strong>Python bindings</strong></a>
</td>
</tr>
<tr>
<td>

![inspector](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/inspector.gif?raw=true)

</td>
<td>

```python
agent = DefaultAgent(
    LitellmModel(model_name=...),
    LocalEnvironment(),
)
agent.run("Write a sudoku game")
```

</td>
</tr>
</table>

## Let's get started!

**Option 1:** If you just want to try out the CLI (package installed in anonymous virtual environment)

```bash
pip install uv && uvx mini-swe-agent
# or
pip install pipx && pipx ensurepath && pipx run mini-swe-agent
```

**Option 2:** Install CLI & python bindings in current environment

```bash
pip install mini-swe-agent
mini  # run the CLI
```

**Option 3:** Install from source (developer setup)

```bash
git clone https://github.com/SWE-agent/mini-swe-agent.git
cd mini-swe-agent && pip install -e .
mini  # run the CLI
```

Read more in our [documentation](https://mini-swe-agent.com/latest/):

* [Quick start guide](https://mini-swe-agent.com/latest/quickstart/)
* [Using the `mini` CLI](https://mini-swe-agent.com/latest/usage/mini/)
* [Global configuration](https://mini-swe-agent.com/latest/advanced/global_configuration/)
* [Yaml configuration files](https://mini-swe-agent.com/latest/advanced/yaml_configuration/)
* [Power up with the cookbook](https://mini-swe-agent.com/latest/advanced/cookbook/)
* [FAQ](https://mini-swe-agent.com/latest/faq/)
* [Contribute!](https://mini-swe-agent.com/latest/contributing/)

## CI Repair Setup And Run

This repo includes a CI-repair benchmark runner at `src/minisweagent/run/benchmarks/cibench.py`.
The benchmark command supports:

- `baseline`: no memory retrieval
- `L1`: file-level retrieval only
- `L1+L2`: file-level + repo-level retrieval
- `L1+L2+L3`: file-level + repo-level + cross-repo retrieval

### Project Setup Commands

If you want to run the benchmark from the Python scripts in this repo, use this setup flow.

#### 1. Install the project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

#### 2. Create project `.env`

The benchmark wrapper and Python scripts now read the repo-level `.env` automatically.

```bash
cat > .env <<'EOF'
MINIMAX_API_KEY=<your_openrouter_key>
OPENROUTER_API_KEY=<your_openrouter_key>
MINIMAX_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MEMCI_LLM_MODEL=minimax/minimax-m2.5
EOF
```

Important:

- Use `minimax/minimax-m2.5`, not `MiniMax-M2.5`
- Shell environment variables still override `.env` if you set them explicitly

#### 3. Prepare the benchmark data

If you already have the TRS export from `CI-REPAIR-BENCH`, copy it in:

```bash
python scripts/copy_trs_data.py
```

If you want to regenerate the split yourself instead:

```bash
python scripts/prepare_shared_dataset.py \
  --trs-dir ~/Documents/CI-REPAIR-BENCH/results/trs_split \
  --parquet ~/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet
```

#### 4. Seed shared memory

Fast heuristic seeding:

```bash
python scripts/seed_memory.py --no-llm-analysis
```

LLM-backed seeding with MiniMax:

```bash
python scripts/seed_memory.py
```

#### 5. Run the eval issues

Wrapper script:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline
bash scripts/run_cibench_minimax_openrouter.sh --ablation L1
bash scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2
bash scripts/run_cibench_minimax_openrouter.sh
```

Direct Python entry point:

```bash
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_dataset.jsonl \
  --output results/l1_l2_l3 \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml \
  --context-model "${MEMCI_LLM_MODEL:-minimax/minimax-m2.5}" \
  --memory-enabled \
  --memory-root results/shared_memory \
  --memory-ablation L1+L2+L3
```

Quick smoke-test on a small slice:

```bash
bash scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:10
```

### Directory Structure

All data, results, and memory files live under the project root:

```
mini-swe-agent-ci-based/
│
├── data/
│   ├── eval_dataset.jsonl        ← 189 eval issues (from TRS split — shared benchmark)
│   ├── memory_seed.jsonl         ← 103 memory issues with ground-truth diffs (for seeding)
│   └── ci_dataset.jsonl          ← full HuggingFace dataset (optional alternative)
│
├── results/
│   ├── shared_memory/            ← L1 / L2 / L3 memory bank (seeded once, used by all agents)
│   │   ├── failure_memory.json   ← L1: per-file failure records
│   │   ├── repo_memory.json      ← L2: repo-level recurring patterns
│   │   └── cross_memory.json     ← L3: cross-repo generalised principles
│   │
│   ├── baseline/                 ← output of the no-memory run
│   │   ├── preds.json            ← predicted patch per instance {id, sha_fail, diff}
│   │   ├── cibench.log           ← full run log
│   │   └── <instance_id>/
│   │       └── <instance_id>.traj.json   ← per-instance agent trajectory
│   │
│   ├── l1/                       ← output of L1-only retrieval run
│   ├── l1_l2/                    ← output of L1+L2 retrieval run
│   └── l1_l2_l3/                 ← output of L1+L2+L3 retrieval run
│
└── scripts/
    ├── prepare_shared_dataset.py           ← joins TRS split + parquet → eval + memory JSONL
    ├── seed_memory.py                      ← seeds results/shared_memory/ from memory_seed.jsonl
    ├── fetch_dataset.py                    ← downloads full dataset from HuggingFace
    └── run_cibench_minimax_openrouter.sh   ← convenience runner for MiniMax via OpenRouter
```

Create the directories once:

```bash
mkdir -p data results/shared_memory results/baseline results/l1 results/l1_l2 results/l1_l2_l3
```

### File Reference

| File | What it contains |
|------|-----------------|
| `data/eval_dataset.jsonl` | **189 eval issues** — the shared TRS benchmark every agent runs on |
| `data/memory_seed.jsonl` | **103 memory issues** with ground-truth diffs — fed into `seed_memory.py` |
| `data/ci_dataset.jsonl` | Full HuggingFace dataset (all instances, alternative to TRS split) |
| `results/shared_memory/failure_memory.json` | **L1** — per-file records: file path, error type, failure pattern, fix direction |
| `results/shared_memory/repo_memory.json` | **L2** — repo-level patterns: aggregated per-file entries, overall failure reason, fix approach |
| `results/shared_memory/cross_memory.json` | **L3** — cross-repo principles: abstract patterns merged across repos by (error_type, issue_type) |
| `results/<run>/preds.json` | Final predictions: `{"<instance_id>": {"id": ..., "sha_fail": ..., "diff": ...}}` |
| `results/<run>/<id>/<id>.traj.json` | Full agent trajectory for one instance (messages, tool calls, exit status) |
| `results/<run>/cibench.log` | Timestamped log of the entire benchmark run |

### Dataset Fields

Each line in `ci_dataset.jsonl` must be a JSON object with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `instance_id` | `str` | ✅ | Unique identifier, e.g. `"owner-repo-abc123"` |
| `sha_fail` | `str` | ✅ | Git commit SHA where CI failed |
| `repo_owner` | `str` | ✅ | GitHub organisation or user name |
| `repo_name` | `str` | ✅ | Repository name (without owner) |
| `workflow_path` | `str` | ✅ | Path to the workflow file, e.g. `".github/workflows/test.yml"` |
| `workflow_name` | `str` | ✅ | Human-readable workflow name, e.g. `"test"` |
| `workflow` | `str` | ✅ | Full YAML text of the workflow file |
| `logs` | `str` or `list` | ✅ | Raw CI logs — either a plain string or a list of `{"step_name": "...", "log": "..."}` |

### Prerequisites

- Python 3.10+
- Git available in `PATH`
- An LLM API key configured for the model you want to use through LiteLLM
- `datasets` and `huggingface_hub` packages for fetching the dataset

### Install

Use a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate

# Upgrade pip (fixes SSL issues on macOS + pyenv)
pip install --upgrade pip \
  --trusted-host pypi.org \
  --trusted-host pypi.python.org \
  --trusted-host files.pythonhosted.org

pip install -e .
```

If you want local dense retrieval for memory, install at least one embedding backend:

```bash
pip install sentence-transformers   # recommended
# or
pip install fastembed               # lighter alternative
```

Install dataset utilities:

```bash
pip install datasets huggingface_hub
```

### Fetch the Dataset

Download CI-REPAIR-BENCH from HuggingFace into `data/ci_dataset.jsonl`:

```bash
# Fetch the test split (default)
python scripts/fetch_dataset.py

# Fetch a specific split
python scripts/fetch_dataset.py --split train

# Fetch all splits into one file
python scripts/fetch_dataset.py --split all

# Custom output path
python scripts/fetch_dataset.py --out data/my_subset.jsonl --split test
```

Or directly in Python:

```python
from datasets import load_dataset
import json, pathlib

ds = load_dataset("ci-benchmark-user/ci-repair-bench", split="test")

out = pathlib.Path("data/ci_dataset.jsonl")
out.parent.mkdir(exist_ok=True)
with out.open("w") as fh:
    for row in ds:
        fh.write(json.dumps(dict(row)) + "\n")

print(f"Saved {len(ds)} instances → {out}")
```

### Shared Benchmark Setup (run ONCE)

For consistent, fair comparison across agents (mini-swe-agent, mem-ci-repair-agent, etc.)
use the **TRS split** from CI-REPAIR-BENCH: 103 memory issues (30%) and 189 eval issues (70%),
selected by recurrence + temporal ordering so every agent sees identical prior knowledge and
identical problems.

The pre-computed TRS data already lives in `CI-REPAIR-BENCH/baselines/results/trs/`.
Copy it directly into this project with a single command — no recomputation needed.

#### Step 1 — Copy TRS data (primary path)

```bash
# Auto-detects CI-REPAIR-BENCH as a sibling or ~/Documents/CI-REPAIR-BENCH
python scripts/copy_trs_data.py

# Explicit paths
python scripts/copy_trs_data.py \
    --trs-results ~/Documents/CI-REPAIR-BENCH/baselines/results/trs \
    --parquet     ~/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet
```

This produces:

| File | Contents |
|------|----------|
| `data/eval_dataset.jsonl` | **189 eval issues** — the benchmark every agent is evaluated on |
| `results/shared_memory/failure_memory.json` | **L1** — 136 per-file failure records (copied directly) |
| `results/shared_memory/repo_memory.json` | **L2** — 80 repo-level recurring patterns (copied directly) |
| `results/shared_memory/cross_memory.json` | **L3** — 66 cross-repo generalised principles (copied directly) |

The eval issues are enriched with all TRS pipeline artifacts — `overall_failure_reasons`,
`effected_files`, `failed_jobs`, `top_memory_peers`, `rcss_score` — so the agent can
**skip LLM-based log analysis** entirely (saves ~567 LLM calls across 189 issues).

> **Alternative: regenerate from raw split + parquet**
>
> If you want to rebuild from scratch rather than reuse the pre-computed data:
> ```bash
> python scripts/prepare_shared_dataset.py \
>     --trs-dir  ~/Documents/CI-REPAIR-BENCH/results/trs_split \
>     --parquet  ~/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet
> python scripts/seed_memory.py --no-llm-analysis   # fast heuristic seeding
> # or with LLM (richer L1/L2/L3):
> # export MINIMAX_API_KEY=<key>; python scripts/seed_memory.py
> ```

#### Step 2 — Set your API key

```bash
export MINIMAX_API_KEY=<your_openrouter_key>
```

#### Step 3 — Run all four ablations

```bash
# 1. Baseline — no memory
scripts/run_cibench_minimax_openrouter.sh --ablation baseline

# 2. L1 — file-level memory only
scripts/run_cibench_minimax_openrouter.sh --ablation L1

# 3. L1+L2 — file-level + repo-level memory
scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2

# 4. L1+L2+L3 — full hierarchical memory (default)
scripts/run_cibench_minimax_openrouter.sh
```

> **Why this matters:** every agent that uses `copy_trs_data.py` from the same
> CI-REPAIR-BENCH TRS export gets **identical eval issues** and **identical starting memory**
> — so differences in results reflect only the agent's repair capability.

---

### Basic Run Command

Both `mini` and `mini-swe-agent` point to the same CLI. The examples below use `mini-swe-agent`.

```bash
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_dataset.jsonl \
  --output results/run_01 \
  -m openai/gpt-4.1 \
  --context-model gpt-4o-mini
```

Useful flags:

- `--filter "^repo_or_instance_regex"`: run only matching instances
- `--slice 0:10`: run a subset
- `--shuffle`: shuffle before slicing
- `--redo-existing`: rerun instances already present in `preds.json`
- `--memory-top-k 3`: top-k retrieval per memory level
- `--save-memory` / `--no-save-memory`: control whether successful fixes are written back to memory

### Running Different LLMs

The CI repair runner already supports swapping the agent model per run with `-m/--model`.
Examples:

```bash
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_dataset.jsonl \
  --output results/claude \
  -m anthropic/claude-sonnet-4-5-20250929 \
  --context-model gpt-4o-mini
```

```bash
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_dataset.jsonl \
  --output results/gpt41 \
  -m openai/gpt-4.1 \
  --context-model gpt-4o-mini
```

### Multi-Model Agent Runs

For the main repair agent, this project can already run multiple models through the meta-model classes:

- `RouletteModel`: randomly chooses one configured model at each agent turn
- `InterleavingModel`: alternates between configured models in a fixed sequence

An example config is included at:

- `src/minisweagent/config/benchmarks/cibench_multi_model_example.yaml`

Run it like this:

```bash
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_dataset.jsonl \
  --output results/multi_model \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_multi_model_example.yaml \
  --context-model gpt-4o-mini
```

Important note:

- The main agent model can be multi-model through the config above.
- `--context-model` is still a single model string for CI log analysis and workflow analysis in `cibench`.

### MiniMax M2.5

This repo now includes a ready OpenRouter-based MiniMax M2.5 setup, which matches the environment variables below.

If your actual setup is through OpenRouter with these environment variables:

```bash
export MINIMAX_API_KEY='your_key_here'
export MINIMAX_BASE_URL='https://openrouter.ai/api/v1'
export MEMCI_LLM_MODEL='minimax/minimax-m2.5'
```

you can run directly with the wrapper script added to this repo:

```bash
# Full run with memory (default — L1+L2+L3)
scripts/run_cibench_minimax_openrouter.sh

# Baseline (no memory)
scripts/run_cibench_minimax_openrouter.sh --ablation baseline
```

This script:

- reads `data/eval_dataset.jsonl` and `results/shared_memory/` by default
- maps `MINIMAX_API_KEY` to `OPENROUTER_API_KEY`
- uses `src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml`
- passes `MEMCI_LLM_MODEL` into `--context-model`

Direct CLI equivalent:

```bash
OPENROUTER_API_KEY="$MINIMAX_API_KEY" \
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_dataset.jsonl \
  --output results/l1_l2_l3 \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml \
  --context-model "${MEMCI_LLM_MODEL:-minimax/minimax-m2.5}" \
  --memory-enabled \
  --memory-root results/shared_memory \
  --memory-ablation L1+L2+L3
```

MiniMax inside a multi-model pool:

```bash
python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_dataset.jsonl \
  --output results/multi_model_minimax \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_multi_model_example.yaml \
  --context-model minimax/minimax-m2.5
```

Before using the example file, replace:

- `YOUR_MINIMAX_API_KEY`

### Baseline And Ablation Commands

All four runs share the same 189-issue dataset (`data/eval_dataset.jsonl`) and the same
pre-seeded memory bank (`results/shared_memory/`).
Run them in order — baseline first, then L1, L1+L2, L1+L2+L3 — for a clean ablation.

#### Prerequisites

```bash
# 1. Copy TRS data (once)
python scripts/copy_trs_data.py

# 2. Set API key
export MINIMAX_API_KEY=<your_openrouter_key>
```

---

#### Run 1 — Baseline (no memory)

The agent receives the eval issue (logs, workflow, repo context) but **no retrieved memory**.
Used as the zero-memory performance floor.

**Convenience script:**
```bash
scripts/run_cibench_minimax_openrouter.sh --ablation baseline
# output → results/baseline/
```

**Direct CLI equivalent:**
```bash
OPENROUTER_API_KEY="$MINIMAX_API_KEY" \
python -m minisweagent.run.benchmarks.cibench \
  --dataset    data/eval_dataset.jsonl \
  --output     results/baseline \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml \
  --context-model "${MEMCI_LLM_MODEL:-minimax/minimax-m2.5}" \
  --no-memory-enabled \
  --save-memory \
  --memory-root results/shared_memory
```

---

#### Run 2 — L1 (file-level memory)

Retrieves **per-file failure records** from `failure_memory.json`.
The agent sees past fixes for the same files that are failing.

**Convenience script:**
```bash
scripts/run_cibench_minimax_openrouter.sh --ablation L1
# output → results/l1/
```

**Direct CLI equivalent:**
```bash
OPENROUTER_API_KEY="$MINIMAX_API_KEY" \
python -m minisweagent.run.benchmarks.cibench \
  --dataset    data/eval_dataset.jsonl \
  --output     results/l1 \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml \
  --context-model "${MEMCI_LLM_MODEL:-minimax/minimax-m2.5}" \
  --memory-enabled \
  --memory-root results/shared_memory \
  --memory-ablation L1
```

---

#### Run 3 — L1+L2 (file-level + repo-level memory)

Adds **repo-level recurring patterns** from `repo_memory.json` on top of L1.
The agent sees both file-specific history and overall patterns for the same repository.

**Convenience script:**
```bash
scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2
# output → results/l1_l2/
```

**Direct CLI equivalent:**
```bash
OPENROUTER_API_KEY="$MINIMAX_API_KEY" \
python -m minisweagent.run.benchmarks.cibench \
  --dataset    data/eval_dataset.jsonl \
  --output     results/l1_l2 \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml \
  --context-model "${MEMCI_LLM_MODEL:-minimax/minimax-m2.5}" \
  --memory-enabled \
  --memory-root results/shared_memory \
  --memory-ablation L1+L2
```

---

#### Run 4 — L1+L2+L3 (full hierarchical memory — default)

Adds **cross-repo generalised principles** from `cross_memory.json` on top of L1+L2.
The agent also sees abstract fix patterns merged across all repositories.

**Convenience script:**
```bash
scripts/run_cibench_minimax_openrouter.sh          # --ablation L1+L2+L3 is the default
# output → results/l1_l2_l3/
```

**Direct CLI equivalent:**
```bash
OPENROUTER_API_KEY="$MINIMAX_API_KEY" \
python -m minisweagent.run.benchmarks.cibench \
  --dataset    data/eval_dataset.jsonl \
  --output     results/l1_l2_l3 \
  -c src/minisweagent/config/benchmarks/cibench.yaml \
  -c src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml \
  --context-model "${MEMCI_LLM_MODEL:-minimax/minimax-m2.5}" \
  --memory-enabled \
  --memory-root results/shared_memory \
  --memory-ablation L1+L2+L3
```

---

### Quick Reference: All Four Runs

```bash
export MINIMAX_API_KEY=<your_key>

scripts/run_cibench_minimax_openrouter.sh --ablation baseline   # → results/baseline/
scripts/run_cibench_minimax_openrouter.sh --ablation L1         # → results/l1/
scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2      # → results/l1_l2/
scripts/run_cibench_minimax_openrouter.sh                       # → results/l1_l2_l3/
```

Useful extra flags (pass after `--ablation`):

| Flag | Effect |
|------|--------|
| `--slice 0:10` | Run only first 10 instances (quick smoke-test) |
| `--filter "^owner/repo"` | Run only instances matching the regex |
| `--redo-existing` | Rerun instances already present in `preds.json` |
| `--memory-top-k 5` | Change retrieval depth (default: 3) |

### Outputs

Each run writes to its output directory:

```
results/<run>/
├── preds.json                         ← predicted patch per instance {id, sha_fail, diff}
├── cibench.log                        ← timestamped log of the full run
└── <instance_id>/
    └── <instance_id>.traj.json        ← full agent trajectory (messages, tool calls, exit status)
```

Memory-enabled runs also update the shared bank in `results/shared_memory/` with any
new successful repairs, so each successive run can build on prior knowledge.

### Algorithm: How the Pipeline Uses TRS Artifacts

The `eval_dataset.jsonl` produced by `copy_trs_data.py` carries pre-computed TRS fields.
The pipeline uses them in two optimisation phases:

**Phase A — Skip LLM log analysis (saves ~567 LLM calls)**

When an instance already has `overall_failure_reasons`, `effected_files`, or `failed_jobs`
from the TRS pipeline, the agent **does not call `CILogAnalyzer`** (which normally costs 3+
LLM calls per issue for chunking + classification). Instead it maps:

| TRS field | Used as |
|-----------|---------|
| `overall_failure_reasons` | `error_context` injected into the repair prompt |
| `effected_files[*]` | `relevant_files` with `file`, `failed_cmd`, `issue_type` |
| `overall_error_types` | `error_types` structured classification |
| `failed_jobs` | `failed_job` list for workflow profile |

**Phase B — Extract validation command from `failed_jobs`**

Instead of regex-parsing the full YAML workflow to guess what command to run,
the agent reads `failed_jobs[*].failed_command` directly — that is the exact
command that broke CI and is therefore the correct validation command to re-run.
This eliminates a class of wrong-command errors and speeds up Phase B.

## Attribution

If you found this work helpful, please consider citing the [SWE-agent paper](https://arxiv.org/abs/2405.15793) in your work:

```bibtex
@inproceedings{yang2024sweagent,
  title={{SWE}-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={John Yang and Carlos E Jimenez and Alexander Wettig and Kilian Lieret and Shunyu Yao and Karthik R Narasimhan and Ofir Press},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/abs/2405.15793}
}
```

Our other projects:

<div align="center">
  <a href="https://github.com/SWE-agent/SWE-agent"><img src="https://raw.githubusercontent.com/SWE-agent/swe-agent-media/refs/heads/main/media/logos_banners/sweagent_logo_text_below.svg" alt="SWE-agent" height="120px"></a>
   &nbsp;&nbsp;
  <a href="https://github.com/SWE-agent/SWE-ReX"><img src="https://raw.githubusercontent.com/SWE-agent/swe-agent-media/refs/heads/main/media/logos_banners/swerex_logo_text_below.svg" alt="SWE-ReX" height="120px"></a>
   &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/SWE-bench"><img src="https://raw.githubusercontent.com/SWE-agent/swe-agent-media/refs/heads/main/media/logos_banners/swebench_logo_text_below.svg" alt="SWE-bench" height="120px"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/SWE-smith"><img src="https://raw.githubusercontent.com/SWE-agent/swe-agent-media/refs/heads/main/media/logos_banners/swesmith_logo_text_below.svg" alt="SWE-smith" height="120px"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/codeclash-ai/codeclash"><img src="https://raw.githubusercontent.com/SWE-agent/swe-agent-media/refs/heads/main/media/logos_banners/codeclash_logo_text_below.svg" alt="CodeClash" height="120px"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/sb-cli"><img src="https://raw.githubusercontent.com/SWE-agent/swe-agent-media/refs/heads/main/media/logos_banners/sbcli_logo_text_below.svg" alt="sb-cli" height="120px"></a>
</div>
