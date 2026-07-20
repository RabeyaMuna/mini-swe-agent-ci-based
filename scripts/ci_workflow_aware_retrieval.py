"""
CI workflow validation-sequence extraction.

This module has one job:

1. Receive the GitHub Actions workflow YAML from the benchmark instance.
2. Ask an LLM which repo files are required to understand the workflow's
   validation commands, for example .pre-commit-config.yaml, reusable
   workflows, or local composite actions.
3. Read those files from the checkout when available.
4. Ask the LLM to return the ordered validation sequence that GitHub Actions
   runs, using only the workflow YAML and dependent file contents.

No hardcoded stage guessing is used here.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import demjson3  # type: ignore
except Exception:
    demjson3 = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class WorkflowValidationExtractionError(RuntimeError):
    """Raised when CI workflow validation sequence extraction fails."""


STRICT_JSON_OBJECT_RULES = """### Output Rules (STRICT)
- Output MUST be a single raw JSON object.
- Do NOT wrap the JSON in triple backticks.
- Do NOT include ```json or any other marker/fence.
- Use double quotes for every JSON key and string value.
- Do not emit trailing commas.
- Do NOT add any text before or after the JSON."""

STRICT_JSON_ARRAY_RULES = """### Output Rules (STRICT)
- Output MUST be a single raw JSON array.
- Do NOT wrap the JSON in triple backticks.
- Do NOT include ```json or any other marker/fence.
- Use double quotes for every JSON key and string value.
- Do not emit trailing commas.
- Do NOT add any text before or after the JSON."""


def _call_llm(llm: Any, prompt: str) -> str:
    if llm is None:
        raise WorkflowValidationExtractionError(
            "LLM is required for workflow validation extraction."
        )
    try:
        result = llm.invoke(prompt)
        return str(getattr(result, "content", result) or "").strip()
    except AttributeError:
        result = llm(prompt)
        return str(getattr(result, "content", result) or "").strip()


def _load_json(content: str, default: Any) -> Any:
    content = str(content or "").strip()

    # Strip markdown fences (```json ... ``` or ``` ... ```)
    if content.startswith("```"):
        lines = content.splitlines()
        # Remove first line (```json or ```)
        if lines:
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # Try direct JSON parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError as json_err:
        # If direct parse fails, try to extract JSON from text
        # Look for JSON array [...] or object {...}
        extracted = None

        # Try to find JSON array
        array_match = re.search(r"\[[\s\S]*\]", content)
        if array_match:
            extracted = array_match.group(0)

        # If no array found, try to find JSON object
        if not extracted:
            obj_match = re.search(r"\{[\s\S]*\}", content)
            if obj_match:
                extracted = obj_match.group(0)

        # Try parsing the extracted JSON
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass  # Fall through to demjson3

        # Last resort: try demjson3
        demjson3_err: Any = "demjson3 is not installed"
        try:
            if demjson3 is not None:
                # Try on original content first
                try:
                    return demjson3.decode(content)
                except Exception:
                    # If that fails and we extracted something, try on extracted
                    if extracted:
                        return demjson3.decode(extracted)
                    raise
        except Exception as exc:
            demjson3_err = exc

        LOGGER.warning(
            "Workflow JSON parse failed: json=%s; demjson3=%s",
            json_err,
            demjson3_err,
        )
        return default


def _read_repo_file(
    repo_path: Optional[str], rel_path: str, max_chars: int = 80_000
) -> Optional[str]:
    """Read file from repo working tree (current checkout state)."""
    if not repo_path or not rel_path:
        return None

    root = Path(repo_path).resolve()
    path = (root / rel_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None

    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return None


def _read_repo_file_at_commit(
    repo_path: Optional[str],
    rel_path: str,
    sha: Optional[str] = None,
    max_chars: int = 80_000,
) -> Optional[str]:
    """
    Read file from repo at specific commit without checking out.

    Uses `git show {sha}:{path}` to read file content at a specific commit
    without modifying the working tree. Falls back to working tree if no SHA.

    Args:
        repo_path: Path to git repository
        rel_path: Relative path to file within repo
        sha: Git commit SHA (if None, reads from working tree)
        max_chars: Maximum characters to read

    Returns:
        File content or None if file doesn't exist or read fails
    """
    if not repo_path or not rel_path:
        return None

    # Normalize path (remove leading /)
    rel_path = str(rel_path).lstrip("/")

    # If no SHA specified, use current working tree
    if not sha:
        return _read_repo_file(repo_path, rel_path, max_chars)

    # Read from specific commit using git show
    import subprocess

    try:
        result = subprocess.run(
            ["git", "show", f"{sha}:{rel_path}"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,  # Prevent hanging
        )
        content = result.stdout[:max_chars]
        return content
    except subprocess.CalledProcessError:
        # File doesn't exist at that commit or git command failed
        return None
    except subprocess.TimeoutExpired:
        LOGGER.warning(f"Timeout reading {rel_path} at {sha}")
        return None
    except Exception as e:
        LOGGER.warning(f"Failed to read {rel_path} at {sha}: {e}")
        return None


def _normalize_dependent_files(raw: Any) -> List[Dict[str, str]]:
    payload = raw if isinstance(raw, dict) else {}
    rows = payload.get("dependent_files") or []
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("file") or "").strip().lstrip("/")
        reason = str(item.get("reason") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append({"path": path, "reason": reason})
    return out


def _normalize_validation_sequence(raw_steps: Any) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for index, item in enumerate(
        raw_steps if isinstance(raw_steps, list) else [], start=1
    ):
        if not isinstance(item, dict):
            continue

        installation_cmd = str(item.get("installation_cmd") or "").strip()
        validation_cmd = str(
            item.get("validation_cmd") or item.get("command") or ""
        ).strip()
        validates = str(item.get("validates") or item.get("name") or "").strip()
        source = str(item.get("source") or item.get("source_file") or "").strip()
        evidence = str(item.get("evidence") or "").strip()

        if not installation_cmd and not validation_cmd:
            continue

        steps.append(
            {
                "order": int(item.get("order") or index),
                "validates": validates,
                "installation_cmd": installation_cmd,
                "validation_cmd": validation_cmd,
                "source": source,
                "evidence": evidence,
            }
        )

    steps.sort(key=lambda row: int(row["order"]))
    for index, step in enumerate(steps, start=1):
        step["order"] = index
    return steps


def build_dependent_file_prompt(workflow_path: str, workflow_content: str) -> str:
    return f"""You are analyzing a GitHub Actions workflow from a benchmark instance.

Your task is to identify ALL repo files that are required to understand the workflow's
actual validation commands.

Include a file when the workflow references or depends on it. Scan for:

1. **Config files** that define validation tools/hooks:
   - ".pre-commit-config.yaml" (pre-commit hooks)
   - "pyproject.toml" (tool configurations)
   - "tox.ini", "setup.cfg", ".flake8", etc.

2. **Reusable workflows and actions**:
   - ".github/workflows/*.yml" (reusable workflows)
   - ".github/actions/*/action.yml" (local composite actions)

3. **Scripts referenced in run: commands**:
   - Shell scripts: "./scripts/validate.sh", "./bin/check.sh"
   - Python scripts: "python scripts/lint.py"
   - Any custom executable referenced with explicit path
   - Look for: "run: ./", "run: scripts/", "run: bin/", "run: python "

4. **Validation config files explicitly referenced**:
   - Files passed as arguments: "--config myconfig.yml"
   - Files in working-directory paths

IMPORTANT: Include the FULL PATH as it appears in the workflow (e.g., "scripts/validate.sh" not just "validate.sh")

WORKFLOW PATH
{workflow_path}

WORKFLOW YAML FROM BENCHMARK
{workflow_content}

{STRICT_JSON_OBJECT_RULES}

Return this JSON object:
{{
  "dependent_files": [
    {{
      "path": ".pre-commit-config.yaml",
      "reason": "workflow runs pre-commit, hooks define the validations"
    }}
  ]
}}

If no dependent files are needed, return:
{{"dependent_files": []}}
"""


def build_validation_sequence_prompt(
    workflow_path: str,
    workflow_content: str,
    dependent_file_contents: List[Dict[str, Any]],
) -> str:
    return f"""You are extracting the ordered CI validation sequence from a GitHub Actions workflow.

Use ONLY:
1. The benchmark workflow YAML.
2. The dependent file contents provided below.

Do not guess or invent commands. Every command must be directly supported by
workflow YAML or dependent file content. Preserve the order GitHub Actions runs
the validations in.

For each validation step:
- "order": execution order.
- "validates": what this step validates, based on the step name and command.
- "installation_cmd": exact install/setup command if the step installs dependencies, else "".
- "validation_cmd": exact validation command if the step validates/lints/tests/builds, else "".
- "source": workflow path or dependent file path where the command comes from.
- "evidence": short evidence from the workflow/dependent file.

For pre-commit:
- Include the workflow command, such as "pre-commit run --all-files".
- Use .pre-commit-config.yaml only to explain what validations that command runs.
- Do not invent hook commands that are not present in the config.

For reusable workflows or local composite actions:
- Follow the referenced file content if it was provided.
- Keep the final order consistent with how the parent workflow calls it.

WORKFLOW PATH
{workflow_path}

WORKFLOW YAML FROM BENCHMARK
{workflow_content}

DEPENDENT FILE CONTENTS
{json.dumps(dependent_file_contents, indent=2, ensure_ascii=False)}

{STRICT_JSON_ARRAY_RULES}

Return this JSON array:
[
  {{
    "order": 1,
    "validates": "Dependencies installation",
    "installation_cmd": "uv install",
    "validation_cmd": ""
  }},
  {{
    "order": 2,
    "validates": "Code style and imports",
    "installation_cmd": "",
    "validation_cmd": "ruff check"
  }},
  // ... additional steps as needed
]
"""


def analyze_workflow_from_benchmark(
    *,
    workflow_content: str,
    workflow_path: str = "",
    repo_path: Optional[str] = None,
    llm: Any = None,
    issue_id: str = "",
    sha_fail: str = "",
) -> Dict[str, Any]:
    """Extract ordered CI validation sequence from benchmark workflow content."""
    workflow_content = str(workflow_content or "")
    workflow_path = str(workflow_path or "")
    issue_id = str(issue_id or "")
    sha_fail = str(sha_fail or "")
    if not workflow_content.strip():
        raise WorkflowValidationExtractionError("workflow_content is required.")

    dependent_raw = _call_llm(
        llm, build_dependent_file_prompt(workflow_path, workflow_content)
    )
    dependent_files = _normalize_dependent_files(
        _load_json(dependent_raw, {"dependent_files": []})
    )

    # Read dependent files from the failing commit (sha_fail)
    # This ensures we analyze the exact state that caused the CI failure
    if sha_fail:
        print(f"       Reading dependent files from commit: {sha_fail[:8]}...")

    dependent_file_contents: List[Dict[str, Any]] = []
    for dep in dependent_files:
        # Use sha_fail to read from the exact failing commit
        content = _read_repo_file_at_commit(
            repo_path, dep["path"], sha=sha_fail or None
        )

        found = content is not None
        if sha_fail:
            status = "✓" if found else "✗"
            print(f"         {status} {dep['path']}")

        dependent_file_contents.append(
            {
                "path": dep["path"],
                "reason": dep.get("reason", ""),
                "found": found,
                "content": content or "",
            }
        )

    sequence_raw = _call_llm(
        llm,
        build_validation_sequence_prompt(
            workflow_path, workflow_content, dependent_file_contents
        ),
    )

    # Log raw LLM response for debugging
    print("[DEBUG] Workflow validation LLM raw response (first 1000 chars):")
    print(f"{str(sequence_raw)[:1000]}")
    print(f"[DEBUG] Response length: {len(str(sequence_raw))} chars")

    validation_sequence = _normalize_validation_sequence(_load_json(sequence_raw, []))

    if not validation_sequence:
        raise WorkflowValidationExtractionError(
            "Failed to extract CI workflow validation sequence from workflow/dependent files."
        )

    result = {
        "id": issue_id,
        "sha_fail": sha_fail,
        "workflow_path": workflow_path,
        "dependent_files": dependent_files,
        "validation_sequence": validation_sequence,
    }
    return result
