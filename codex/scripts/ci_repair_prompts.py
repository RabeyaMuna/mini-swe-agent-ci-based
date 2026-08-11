"""Cache-friendly prompt layout for the CI repair agents.

OpenAI prompt caching matches exact prompt prefixes.  Keep instructions in the
stable prefix and append repository/issue/problem data after the explicit
boundary.  The dynamic suffix is never included in the template fingerprint.
"""

from __future__ import annotations

import hashlib
from typing import Any


DYNAMIC_CONTEXT_BOUNDARY = "\n<!-- ci-repair-dynamic-context:v1 -->\n"

_COMMON_REPAIR_INSTRUCTIONS = r"""# CI Repair Task

You are fixing one CI failure problem in the current repository. Perform the
full analysis needed for a correct repair; do not omit relevant configuration,
dependency, package, environment, or source-code evidence from the dynamic
context supplied below.

## Instructions

**Your Task:**
- Analyze the supplied CI failure context
- Identify the root cause from error signals and relevant files
- Determine the minimal fix needed
- Implement and validate the fix

**Evidence and repair quality:**
- Treat every issue as dynamic; derive the repair from the current repository,
  CI evidence, environment constraints, and supplied problem relationships
- Preserve and inspect all relevant configuration changes instead of reducing
  the problem to source-code edits alone
- When a package, dependency declaration, lockfile, resolver option, runtime,
  build tool, or CI image changed, analyze why that change may be required by
  the project's supported versions and environment
- Consider package deprecation, removed APIs, incompatible version ranges,
  platform/runtime support, resolver behavior, build isolation, and the next
  validation step when the available evidence supports those possibilities
- Distinguish observed evidence from a plausible inference; confirm an
  inference in repository files or tool output before changing the project
- Keep related configuration, dependency, and source changes together when
  they form one repair, but do not combine unrelated problems
- Do not discard a configuration or package change merely because the first
  visible CI failure can be fixed elsewhere
- After the first failure is repaired, consider whether the same environment
  constraints expose a directly dependent installation, build, test, lint, or
  runtime failure; validate only what is relevant to this problem
- Prefer the smallest complete repair over a partial edit that hides the first
  error while leaving its supported cause unresolved
- Do not invent package failures or configuration requirements that are not
  supported by the repository, CI logs, dependency metadata, or tool output

**STEP 0: Check if problem still exists (REQUIRED)**
Before attempting any fix:
1. **Verify file paths** - Sometimes CI logs provide incomplete/partial paths:
   - If a file path doesn't exist, search for it: `find . -type f -name "$(basename <file>)"`
   - Use the correct full path for all subsequent commands
2. Run the verification command on ONLY the affected files (not the whole repo)
3. Check if the specific error signals are still present
4. If the errors are NOT found:
   - Report "Problem already fixed by previous step"
   - Commit any staged changes with message "Skip: problem already fixed"
   - Exit successfully
5. If errors ARE found: Proceed to fix

Example for mypy errors:
```bash
# Check only affected files (find correct path if needed)
FILE="path/to/affected_file.py"
if [ ! -f "$FILE" ]; then
  FILE=$(find . -type f -name "$(basename $FILE)" | head -1)
fi
if [ -f "$FILE" ]; then
  mypy "$FILE" 2>&1 | grep -E "line_number"
  # If no output -> Problem is fixed, skip to next
fi
```

**For automated tool failures (formatters, linters, type checkers):**
- Prefer running the tool with auto-fix flags: `black .`, `ruff --fix .`, `mypy --install-types`, etc.
- Let the tool fix all affected files at once
- Only manually edit if the tool cannot auto-fix

**General workflow:**
1. **CHECK if problem exists** (see STEP 0 above - REQUIRED)
2. Inspect the repository and understand the problem from the supplied context
3. Make the minimal correct change to fix the issue
4. Do not modify unrelated files
5. Commit your changes with a descriptive message
6. OPTIONAL: Run validation ONLY on files you changed (not the whole repo)
7. The fix should be committed in git (not as uncommitted changes)

**Scope:**
- Fix this problem only (do not fix unrelated issues)
- Preserve existing behavior unless proven wrong by CI
- Do not remove tests or weaken checks
- Do not update dependencies unless required by the fix
- Preserve relevant configuration and package changes required by the fix
- Do NOT run validation on the entire repository (may have unrelated failures)
- If you verify, run validation ONLY on the specific files you changed

**Important:**
- COMMIT your changes (don't leave as uncommitted diff)
- Use descriptive commit messages
- Don't worry if repo-wide validation fails due to other issues
- Your fix should address the specific problem described below

**Final report:**
- Root cause identified
- Files changed
- Whether validation passed on YOUR changes (not whole repo)
- Any remaining risks
"""

_BASELINE_MODE_INSTRUCTIONS = r"""
## Analysis Mode

Analyze the problem from the complete CI failure and verification context
provided below. Use repository evidence to confirm the supplied root cause and
repair details rather than assuming they are complete.
"""

_MEMORY_MODE_INSTRUCTIONS = r"""
## Analysis Mode

The dynamic context may include a repair plan derived from similar successful
fixes. If a repair plan is provided, use it as the primary strategy, while
confirming it against the current repository and CI evidence. Pay attention to
its pitfalls and validation command. If it is missing or incomplete, derive the
minimal correct fix from the problem description, root cause, affected files,
error signals, and verification details.
"""

_DYNAMIC_CONTEXT_INTRO = r"""
## Dynamic Task Context

Everything after the boundary below belongs to the current issue. Treat it as
authoritative task input. It may change independently for every issue and must
never be replaced by context from another repair.
"""

STABLE_PROMPT_PREFIXES = {
    "baseline": (
        _COMMON_REPAIR_INSTRUCTIONS
        + _BASELINE_MODE_INSTRUCTIONS
        + _DYNAMIC_CONTEXT_INTRO
    ).strip(),
    "memory": (
        _COMMON_REPAIR_INSTRUCTIONS
        + _MEMORY_MODE_INSTRUCTIONS
        + _DYNAMIC_CONTEXT_INTRO
    ).strip(),
}


def prompt_mode(ablation: str) -> str:
    """Map an ablation name to its stable prompt family."""
    return "baseline" if str(ablation).lower() == "baseline" else "memory"


def render_ci_repair_prompt(mode: str, dynamic_context: str) -> str:
    """Render a stable exact prefix followed by untouched dynamic context."""
    try:
        stable_prefix = STABLE_PROMPT_PREFIXES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported CI repair prompt mode: {mode}") from exc

    return (
        stable_prefix
        + DYNAMIC_CONTEXT_BOUNDARY
        + str(dynamic_context).strip()
        + "\n"
    )


def prompt_cache_info(prompt: str) -> dict[str, Any]:
    """Describe the exact-prefix boundary without hashing dynamic issue data."""
    stable_prefix, boundary, dynamic_context = str(prompt).partition(
        DYNAMIC_CONTEXT_BOUNDARY
    )
    if not boundary:
        return {
            "layout": "legacy_dynamic_prompt",
            "template_fingerprint": None,
            "stable_prefix_chars": 0,
            "dynamic_context_chars": len(str(prompt)),
        }

    fingerprint = hashlib.sha256(stable_prefix.encode("utf-8")).hexdigest()[:16]
    return {
        "layout": "stable_prefix_dynamic_suffix_v1",
        "template_fingerprint": fingerprint,
        "stable_prefix_chars": len(stable_prefix),
        "dynamic_context_chars": len(dynamic_context),
    }
