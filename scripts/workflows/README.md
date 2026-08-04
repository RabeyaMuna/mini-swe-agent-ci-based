# Workflow Shell Scripts

This folder contains shell entry points that run multi-step workflows.

Run them from the repository root so their repo-relative paths resolve correctly.

## Common Commands

```bash
MODEL=glm5.2 scripts/workflows/run_memory_decompositions.sh
MODEL=glm5.2 scripts/workflows/run_memory_decompositions.sh agno,flower,camel,crewAI
```

## Files

- `run_memory_decompositions.sh`: split Hugging Face data, then run backward and forward decomposition and memory builds.
- `run_new_workflow.sh`: older split-first memory workflow.
- `run_system.sh`: older intelligent workflow for `data/trs`.
- `run_ablation_study.sh`: run memory ablation experiments.
- `run_all_analyses.sh`: run analysis scripts.
- `run_cibench_minimax_openrouter.sh`: run CI bench with MiniMax/OpenRouter settings.
- `validate_setup.sh`: validate expected local files and environment.
