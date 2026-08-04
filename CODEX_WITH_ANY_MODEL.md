# Run Codex with Any Model (Using Existing Scripts)

You **already have** the setup! Your existing `codex/scripts/run_codex_ci_repair.py` works with HuggingFace data.

## ✅ Setup (One Time):

```bash
# 1. Start LiteLLM proxy
./start_litellm_proxy.sh

# 2. Your .env should have:
OPENAI_API_KEY=sk-dummy  # Dummy key for Codex
# Plus your real API keys for the proxy
```

---

## ✅ Run Codex with Any Model:

### Using Existing Script:

```bash
# GLM 5.2, BASELINE
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations baseline \
    --codex-command "codex exec --full-auto --model glm5.2"

# Minimax, L1+L2+L3 with backward memory
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations L1+L2+L3 \
    --memory-root data/back_trs \
    --codex-command "codex exec --full-auto --model minimax"

# Forward direction (different memory)
PYTHONPATH=. python3 codex/scripts/run_codex_ci_repair.py \
    --issue-ids 125,126,127 \
    --ablations L1+L2+L3 \
    --memory-root data/fwr_trs \
    --codex-command "codex exec --full-auto --model glm5.2"
```

---

## ✅ Results Structure:

Your script saves to:
```
results/codex/{ablation}_{model}/
```

Example:
- `results/codex/baseline_glm5_2/125/`
- `results/codex/l1_l2_l3_minimax/125/`

---

## ✅ Simple Wrapper Script:

Let me create a simple wrapper for you:

```bash
./run_codex_simple.sh <model> <ablation> <direction> <issue-ids>
```

Examples:
```bash
# GLM 5.2, baseline
./run_codex_simple.sh glm5.2 baseline backward 125,126,127

# Minimax, full memory, forward
./run_codex_simple.sh minimax L1+L2+L3 forward 125,126,127
```

---

## ✅ Key Points:

1. **No dataset conversion needed** - Uses HuggingFace directly ✓
2. **Any model** - Just change `--model` parameter ✓
3. **Memory support** - Use `--memory-root` to specify backward/forward ✓
4. **Existing workflow** - Works with your current setup ✓

---

## ✅ Available Models:

All models from `litellm_config.yaml`:
- `glm5.2`
- `glm4`
- `minimax`
- `deepseek-chat`
- `gpt-4`

**Your existing system already supports any model!** 🎉
