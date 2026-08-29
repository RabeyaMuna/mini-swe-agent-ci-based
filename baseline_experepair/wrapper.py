"""
ExpeRepair Baseline Wrapper - No Memory/Iteration

Calls ExpeRepair's write_multiple_patch_wo_memory() method with:
- retries=1 (no iteration)
- patch_nums=1 (single patch)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experepair_core.agents.agent_write_patch import PatchAgent
from experepair_core.data_structures import BugLocation
from experepair_core.post_process import extract_locs_for_files


def generate_patch_experepair_baseline(
    issue_description: str,
    changed_files: list[str],
    repo_path: str,
    model: str = "minimax/minimax-m2.5",
    diff: str = "",
    workflow: str = "",
    validation_commands: str = ""
) -> dict:
    """
    Generate patch using ExpeRepair's approach (no memory, no iteration).

    Args:
        issue_description: CI failure description/logs
        changed_files: List of file paths changed in the commit
        repo_path: Path to repository
        model: LLM model to use
        diff: Git diff (optional, for context)
        workflow: CI workflow YAML (optional, for context)
        validation_commands: Validation sequence (optional, for context)

    Returns:
        dict with 'patch', 'cost', 'model'
    """

    # Create bug locations from changed files
    # (ExpeRepair expects BugLocation objects for patch targets)
    bug_locations = []
    for file_path in changed_files:
        full_path = Path(repo_path) / file_path
        if full_path.exists():
            bug_locations.append(BugLocation(
                file_path=file_path,
                start_line=1,  # Will be refined by ExpeRepair
                end_line=1
            ))

    # Format issue with CI context
    formatted_issue = f"""## CI Failure

{issue_description}

## Workflow Configuration
```yaml
{workflow}
```

## Changed Files
{', '.join(changed_files)}

## Git Diff
```diff
{diff}
```

## Validation Commands
```bash
{validation_commands}
```
"""

    # Initialize ExpeRepair's PatchAgent
    patch_agent = PatchAgent(
        model_name=model,
        backend="openai",  # Use OpenAI-compatible API
        logger=None,
        enable_validation=False,  # No test validation for now
        enable_sbfl=False,  # No spectrum-based fault localization
        enable_angelic=False,  # No angelic debugging
        enable_perfect_angelic=False
    )

    # Call write_multiple_patch_wo_memory with NO iteration
    # retries=1 means single attempt
    # patch_nums=1 means generate only 1 patch
    result = patch_agent.write_multiple_patch_wo_memory(
        test_content=formatted_issue,
        orig_repro_result="",  # No test reproduction needed for CI
        retries=1,  # NO ITERATION
        patch_nums=1  # SINGLE PATCH
    )

    # Extract patch from result
    if result and len(result) > 0:
        patch_data = result[0]  # First (and only) patch
        return {
            "patch": patch_data.get("patch", ""),
            "cost": patch_data.get("cost", 0.0),
            "model": model,
            "method": "experepair_baseline"
        }

    return {
        "patch": "",
        "cost": 0.0,
        "model": model,
        "method": "experepair_baseline"
    }


if __name__ == "__main__":
    # Test the wrapper
    import json

    # Example CI issue
    test_issue = {
        "id": "test_1",
        "repo_name": "test/repo",
        "repo_path": "/tmp/test_repo",
        "failure_description": "ImportError: cannot import name 'ops' from 'ultralytics.yolo.utils'",
        "changed_files": ["wandb/integration/ultralytics/bbox_utils.py"],
        "diff": "diff --git a/file.py b/file.py\n...",
        "workflow": "name: CI\non: [push]\njobs:\n  test:\n    ...",
        "validation_commands": "pytest tests/"
    }

    result = generate_patch_experepair_baseline(
        issue_description=test_issue["failure_description"],
        changed_files=test_issue["changed_files"],
        repo_path=test_issue["repo_path"],
        diff=test_issue.get("diff", ""),
        workflow=test_issue.get("workflow", ""),
        validation_commands=test_issue.get("validation_commands", "")
    )

    print(json.dumps(result, indent=2))
