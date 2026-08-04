#!/bin/bash
# Complete Harbor Setup for Codex Agent + Minimax

set -e

echo "=========================================="
echo "HARBOR SETUP FOR CODEX AGENT"
echo "=========================================="
echo ""

# Step 1: Install Harbor
echo "Step 1/4: Installing Harbor..."
if command -v harbor &> /dev/null; then
    echo "  ✓ Harbor installed: $(harbor --version 2>&1 | head -1)"
else
    echo "  Installing Harbor..."
    pip install harbor-cli
    echo "  ✓ Harbor installed"
fi

echo ""

# Step 2: Activate environment
echo "Step 2/4: Checking Python environment..."
source .venv-codex/bin/activate
pip install datasets python-dotenv pyyaml requests
echo "  ✓ Dependencies ready"

echo ""

# Step 3: Verify .env
echo "Step 3/4: Checking environment variables..."
if [ ! -f .env ]; then
    echo "  ✗ .env file not found"
    echo "  Create it with: echo 'OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY' > .env"
    exit 1
fi

source .env

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "  ✗ OPENROUTER_API_KEY not set in .env"
    echo "  Add it: echo 'OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY' >> .env"
    exit 1
fi

echo "  ✓ OPENROUTER_API_KEY found"

echo ""

# Step 4: Create run script
echo "Step 4/4: Creating run script..."

cat > run_harbor_codex.sh << 'EOFSCRIPT'
#!/bin/bash
# Run Harbor with Codex agent and minimax
#
# Usage: ./run_harbor_codex.sh <issue-ids> [ablation] [direction]

set -e

ISSUE_IDS=${1:-125}
ABLATION=${2:-baseline}
DIRECTION=${3:-backward}

echo "=========================================="
echo "HARBOR + CODEX + MINIMAX"
echo "=========================================="
echo "Issues:    $ISSUE_IDS"
echo "Ablation:  $ABLATION"
echo "Direction: $DIRECTION"
echo ""

# Load environment
[ -f .env ] && source .env

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "✗ OPENROUTER_API_KEY not set"
    exit 1
fi

# Set OpenAI environment for OpenRouter
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=$OPENROUTER_API_KEY

echo "✓ Using OpenRouter"
echo "  Base URL: $OPENAI_BASE_URL"
echo ""

# Convert dataset
echo "Converting issues to Harbor format..."
python3 convert_harbor_dataset.py "$ISSUE_IDS" "$ABLATION" "$DIRECTION"
echo ""

# Run Harbor
echo "Running Harbor with Codex agent..."
echo ""

harbor run \
    --dataset harbor_dataset \
    --agent codex \
    --model openrouter/minimax/minimax-01 \
    --output "results/harbor/${ABLATION}_minimax_${DIRECTION}" \
    --timeout 600

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS!"
    echo "Results: results/harbor/${ABLATION}_minimax_${DIRECTION}/"
else
    echo "✗ FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
EOFSCRIPT

chmod +x run_harbor_codex.sh

# Create dataset converter
cat > convert_harbor_dataset.py << 'EOFPY'
#!/usr/bin/env python3
"""Convert CI-bench to Harbor format with full CI context."""

import json
import sys
from pathlib import Path

def load_hf_dataset():
    from datasets import load_dataset
    ds = load_dataset("ci-benchmark-user/ci-repair-bench", split="train")
    return [dict(item) for item in ds]

def load_cache(path):
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(item.get('id') or item.get('instance_id')): item for item in data}
    return data

def format_ci_markdown(ci):
    md = "\n# CI Failure Analysis\n\n"
    if ci.get("error_context"):
        md += f"## Failure Context\n- {ci['error_context']}\n\n"
    if ci.get("failure_signals"):
        md += "## Failure Signals\n"
        for s in ci["failure_signals"][:5]:
            md += f"- {s}\n"
        md += "\n"
    if ci.get("relevant_files"):
        md += "## Relevant Files\n"
        for f in ci["relevant_files"][:10]:
            md += f"- {f}\n"
        md += "\n"
    return md

def format_verification(ver):
    md = "## CI Verification Details\n"
    for idx, step in enumerate(ver.get("validation_sequence", [])[:5], 1):
        md += f"### Step {idx}: {step.get('description', 'Validation')}\n"
        if step.get("install_command"):
            md += f"- Install: {step['install_command']}\n"
        if step.get("validation_command"):
            md += f"- Validate: {step['validation_command']}\n"
        md += "\n"
    return md

def build_problem_statement(issue, ci, ver):
    files = []
    if ci.get("relevant_files"):
        files = [str(f.get("file") if isinstance(f, dict) else f)
                for f in ci["relevant_files"][:5]]

    signals = ci.get("failure_signals", [])[:5]

    prompt = f"""# CI Repair Task

You are fixing a CI failure in this repository.

## Problem To Fix

**Problem Description:**
{ci.get('error_context', 'Fix CI failure')}

**Affected Files:**
{chr(10).join(f'  - {f}' for f in files) if files else "  (Identify from CI context)"}

**Error Signals:**
{chr(10).join(f'  - {s}' for s in signals) if signals else "  (See CI failure output)"}

**Repository Context:**
- Repository: {issue.get('repo_owner')}/{issue.get('repo_name')}
- Failing commit: {issue.get('sha_fail')}
- Workflow: {issue.get('workflow_path') or issue.get('workflow') or 'unknown'}

{format_ci_markdown(ci)}
{format_verification(ver)}

## Instructions

**Your Task:**
- Analyze the CI failure context
- Identify the root cause
- Implement and validate the fix

**For automated tool failures:**
- Prefer auto-fix: `black .`, `ruff --fix .`, etc.
- Only manually edit if tools can't auto-fix

**Workflow:**
1. Understand the problem from CI context
2. Make minimal correct changes
3. Run validation commands
4. Leave final fix as git diff

**Scope:**
- Fix this problem only
- Preserve existing behavior
- Don't remove tests or weaken checks
"""
    return prompt

def main():
    issue_ids = sys.argv[1].split(',') if len(sys.argv) > 1 else ['125']
    ablation = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
    direction = sys.argv[3] if len(sys.argv) > 3 else 'backward'

    print(f"Loading issues: {', '.join(issue_ids)}")
    all_issues = load_hf_dataset()

    # Load caches
    ci_cache = load_cache(Path("data/log_details.json"))
    ver_cache = load_cache(Path("data/workflow_validation_cache.json"))

    # Filter issues
    selected = [
        issue for issue in all_issues
        if str(issue.get('instance_id') or issue.get('id') or issue.get('sha_fail')) in issue_ids
    ]

    print(f"Found {len(selected)} issues")

    # Create Harbor dataset
    output_dir = Path("harbor_dataset")
    output_dir.mkdir(exist_ok=True)

    with (output_dir / 'instances.jsonl').open('w') as f:
        for issue in selected:
            iid = str(issue.get('instance_id') or issue.get('id'))
            ci = ci_cache.get(iid, {})
            ver = ver_cache.get(iid, {})

            problem = build_problem_statement(issue, ci, ver)

            instance = {
                'id': iid,
                'repo': f"{issue['repo_owner']}/{issue['repo_name']}",
                'base_commit': issue['sha_fail'],
                'problem_statement': problem,
                'metadata': {
                    'sha_fix': issue.get('sha_fix'),
                    'ablation': ablation,
                    'direction': direction,
                }
            }

            f.write(json.dumps(instance) + '\n')
            print(f"  ✓ {iid}")

    print(f"\n✓ Created: {output_dir}/instances.jsonl")

if __name__ == '__main__':
    main()
EOFPY

chmod +x convert_harbor_dataset.py

echo "  ✓ Created run_harbor_codex.sh"
echo "  ✓ Created convert_harbor_dataset.py"

echo ""
echo "=========================================="
echo "✓ HARBOR SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Run Codex with minimax:"
echo ""
echo "  ./run_harbor_codex.sh 125"
echo "  ./run_harbor_codex.sh 125,126,127 L1+L2+L3 backward"
echo ""
