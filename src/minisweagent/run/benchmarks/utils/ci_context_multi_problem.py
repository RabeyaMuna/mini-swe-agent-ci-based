"""
ci_context_multi_problem.py
============================
Enhanced problem statement builder that FORCES the agent to fix ALL atomic problems.

Key Difference from Original:
- Original: Hidden problems shown as "optional guidance" in memory context
- This version: ALL problems presented as REQUIRED, explicit, numbered tasks

Usage:
    Replace build_problem_statement() call with build_problem_statement_multi_problem()
"""

from typing import Any, Dict, List


def build_problem_statement_multi_problem(
    context: Dict[str, Any],
    memory: Dict[str, Any],
) -> str:
    """
    Build problem statement that explicitly lists ALL atomic problems as required tasks.

    Structure:
    1. Header with explicit multi-problem count
    2. Each atomic problem as separate numbered section
    3. Clear verification requirements per problem
    4. Explicit stopping criteria
    5. Full repair sequence checklist
    """

    # Extract atomic problems from memory
    llm_selection = memory.get("llm_selection", {})
    guidance_doc = llm_selection.get("guidance_document", {})

    # Get primary (visible) problem info
    repo = context.get("repo", "")
    sha_fail = context.get("sha_fail", "")
    workflow = context.get("workflow_name") or context.get("workflow_path", "")

    # Extract validation sequence
    profile = context.get("workflow_profile") or {}
    validation_sequence = profile.get("validation_sequence", [])
    validation_cmds = profile.get("validation_cmd", [])

    # Count problems
    primary_files = guidance_doc.get("primary_files", [])
    full_scope = guidance_doc.get("full_scope", {})
    additional_files = full_scope.get("additional_files", [])
    linked_issues = guidance_doc.get("linked_issues", [])

    # Problem 1 is always the visible CI failure
    visible_problem_count = 1

    # Hidden problems come from linked_issues or additional_files
    hidden_problem_count = len(linked_issues)

    # If no linked_issues but we have additional_files, create implicit hidden problem
    if hidden_problem_count == 0 and additional_files:
        hidden_problem_count = 1

    total_problems = visible_problem_count + hidden_problem_count

    # Build header with WARNING
    lines = [
        "# CI REPAIR TASK - MULTI-PROBLEM FAILURE",
        "",
        f"**Repository**: `{repo}`",
        f"**Failing Commit (sha_fail)**: `{sha_fail}`",
        f"**Workflow**: `{workflow}`",
        "",
    ]

    if total_problems > 1:
        lines.extend([
            f"⚠️ **CRITICAL WARNING**: This CI failure contains **{total_problems} DISTINCT PROBLEMS** that MUST ALL be fixed.",
            "",
            "**Why you see only 1 problem in the CI log**:",
            "  - CI workflows stop at the FIRST failure",
            "  - Problems #2, #3, ... are HIDDEN behind Problem #1",
            "  - They WILL cause CI to fail even after you fix Problem #1",
            "  - Historical evidence shows all problems must be fixed together",
            "",
            "**Your task**: Fix ALL problems, not just the first one.",
            "",
            "---",
            "",
        ])
    else:
        lines.extend([
            "**Note**: This appears to be a single-problem CI failure.",
            "",
            "---",
            "",
        ])

    # PROBLEM #1 (VISIBLE) - Always from the CI log
    lines.extend(_build_visible_problem_section(
        context=context,
        guidance_doc=guidance_doc,
        validation_cmds=validation_cmds,
    ))

    # PROBLEM #2, #3, ... (HIDDEN) - From memory guidance
    if linked_issues:
        for idx, linked_issue in enumerate(linked_issues, start=2):
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.extend(_build_hidden_problem_section(
                problem_num=idx,
                linked_issue=linked_issue,
                validation_cmds=validation_cmds,
                validation_sequence=validation_sequence,
            ))
    elif additional_files:
        # Implicit hidden problem from additional_files
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.extend(_build_hidden_problem_from_files(
            problem_num=2,
            additional_files=additional_files,
            guidance_doc=guidance_doc,
            validation_cmds=validation_cmds,
        ))

    # FULL REPAIR SEQUENCE (mandatory checklist)
    if total_problems > 1:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## MANDATORY REPAIR SEQUENCE")
        lines.append("")
        lines.append("You MUST complete ALL steps below:")
        lines.append("")

        for i in range(1, total_problems + 1):
            lines.append(f"**Step {i}**: Fix Problem #{i}")
            if i == 1:
                lines.append(f"   → This is the VISIBLE problem from CI log")
            else:
                lines.append(f"   → This is HIDDEN problem #{i-1} (will fail after Problem #{i-1} is fixed)")

            # Add validation command if available
            if i <= len(validation_cmds):
                lines.append(f"   → Verification: `{validation_cmds[i-1]}`")
            lines.append("")

        lines.append(f"**Step {total_problems + 1}**: Run FULL validation sequence")
        lines.append("   → Ensures all CI stages pass")
        if validation_cmds:
            lines.append("   ```bash")
            for cmd in validation_cmds[:5]:  # Show up to 5 commands
                lines.append(f"   {cmd}")
            lines.append("   ```")
        lines.append("")

        # STOPPING CRITERIA
        lines.append("## STOPPING CRITERIA")
        lines.append("")
        lines.append("❌ **DO NOT STOP** if:")
        lines.append(f"  - You have only fixed Problem #1")
        lines.append(f"  - Only the first validation command passes")
        lines.append(f"  - You have not attempted all {total_problems} problems")
        lines.append("")
        lines.append("✅ **ONLY STOP** when:")
        lines.append(f"  - ALL {total_problems} problems are fixed")
        lines.append(f"  - ALL validation commands pass")
        lines.append("  - The complete CI validation sequence succeeds")
        lines.append("")

        # Evidence from memory
        confidence = guidance_doc.get("confidence", "")
        if confidence:
            lines.append("## WHY ALL PROBLEMS MUST BE FIXED")
            lines.append("")
            lines.append(f"**Confidence Level**: {confidence}")
            lines.append("")
            summary = guidance_doc.get("summary", "")
            if summary:
                lines.append(f"**Evidence**: {summary}")
                lines.append("")

    return "\n".join(lines)


def _build_visible_problem_section(
    context: Dict[str, Any],
    guidance_doc: Dict[str, Any],
    validation_cmds: List[str],
) -> List[str]:
    """Build Problem #1 section (the visible CI failure)."""

    lines = [
        "## PROBLEM #1: VISIBLE IN CI LOG (FIX THIS FIRST)",
        "",
    ]

    # Problem metadata
    error_types = context.get("overall_error_types", [])
    if error_types:
        lines.append(f"**Problem Type**: {', '.join(error_types[:2])}")

    # CI stage
    failed_jobs = context.get("failed_jobs", [])
    if failed_jobs:
        first_job = failed_jobs[0]
        step = first_job.get("step", "")
        cmd = first_job.get("command") or first_job.get("failed_command", "")
        if step:
            lines.append(f"**CI Stage**: {step}")
        if cmd:
            lines.append(f"**Failed Command**: `{cmd}`")

    lines.append("")

    # What's wrong (from CI log analysis)
    failure_reasons = context.get("overall_failure_reasons", [])
    if failure_reasons:
        lines.append("**What Failed (from CI log)**:")
        for reason in failure_reasons[:3]:
            lines.append(f"  - {reason}")
        lines.append("")

    # Diagnosis from memory
    diagnosis = guidance_doc.get("diagnosis", "")
    if diagnosis:
        lines.append("**Root Cause Analysis (from past repairs)**:")
        lines.append(f"  {diagnosis}")
        lines.append("")

    # Primary files to fix
    primary_files = guidance_doc.get("primary_files", [])
    if not primary_files:
        # Fallback to effected_files from context
        primary_files = [
            {"file": f.get("file", ""), "reason": f.get("reason", ""), "fix": ""}
            for f in context.get("effected_files", [])
        ]

    if primary_files:
        lines.append("**Files to Fix**:")
        for pf in primary_files[:10]:  # Limit to 10 files
            if isinstance(pf, dict):
                file_path = pf.get("file", "")
                reason = pf.get("reason", "")
                fix = pf.get("fix", "")

                line = f"  - `{file_path}`"
                if reason:
                    line += f" — {reason}"
                lines.append(line)

                if fix:
                    lines.append(f"    → **Fix**: {fix}")
            else:
                lines.append(f"  - `{pf}`")
        lines.append("")

    # Fix approach
    fix_approach = guidance_doc.get("fix_approach", [])
    if fix_approach:
        lines.append("**How to Fix**:")
        for step in fix_approach[:5]:
            lines.append(f"  - {step}")
        lines.append("")

    # Verification
    if validation_cmds:
        lines.append("**Verification (REQUIRED)**:")
        lines.append("  After fixing this problem, you MUST verify:")
        lines.append("  ```bash")
        lines.append(f"  {validation_cmds[0]}")
        lines.append("  ```")
        lines.append("  Expected: This command should pass ✅")
        lines.append("")

    # Warning about hidden problems
    lines.append("⚠️ **IMPORTANT**: After fixing this problem:")
    lines.append("  - The first CI validation will pass")
    lines.append("  - But CI will proceed to the NEXT validation stage")
    lines.append("  - Problem #2 will appear (if it exists)")
    lines.append("  - DO NOT stop here - continue to Problem #2")

    return lines


def _build_hidden_problem_section(
    problem_num: int,
    linked_issue: Dict[str, Any],
    validation_cmds: List[str],
    validation_sequence: List[Dict[str, Any]],
) -> List[str]:
    """Build Problem #N section (hidden problem from memory)."""

    lines = [
        f"## PROBLEM #{problem_num}: HIDDEN PROBLEM (FIX AFTER PROBLEM #{problem_num-1})",
        "",
        "**Why This is Hidden**:",
        f"  - CI stopped at Problem #{problem_num-1}, never reached this validation stage",
        "  - This problem will ONLY appear after you fix the previous problem(s)",
        "  - But historical repairs show this MUST be fixed too",
        "",
    ]

    # Root cause
    root_cause = linked_issue.get("root_cause", "")
    if root_cause:
        lines.append(f"**Root Cause**: {root_cause}")
        lines.append("")

    # What's wrong
    missing_from_log = linked_issue.get("missing_from_log", "")
    if missing_from_log:
        lines.append("**What Will Fail**:")
        lines.append(f"  {missing_from_log}")
        lines.append("")

    # Workflow stage
    workflow_stage = linked_issue.get("workflow_stage", "")
    if workflow_stage:
        lines.append(f"**CI Validation Stage**: {workflow_stage}")

    # Try to find matching validation command
    cmd_idx = problem_num - 1
    if cmd_idx < len(validation_cmds):
        lines.append(f"**Failed Command**: `{validation_cmds[cmd_idx]}`")
    lines.append("")

    # Affected files
    affected_files = linked_issue.get("affected_files", [])
    fix_pattern = linked_issue.get("fix_pattern", "")

    if affected_files:
        lines.append(f"**Files to Fix** (ALL {len(affected_files)} files):")
        for file_path in affected_files[:15]:  # Limit to 15
            lines.append(f"  - `{file_path}`")
            if fix_pattern:
                lines.append(f"    → **Fix**: {fix_pattern}")

        if len(affected_files) > 15:
            lines.append(f"  - ... and {len(affected_files) - 15} more files")
        lines.append("")

    # Fix pattern
    if fix_pattern and not affected_files:
        lines.append("**How to Fix**:")
        lines.append(f"  {fix_pattern}")
        lines.append("")

    # Verification
    if cmd_idx < len(validation_cmds):
        lines.append("**Verification (REQUIRED)**:")
        lines.append(f"  After fixing Problem #{problem_num}, verify:")
        lines.append("  ```bash")
        lines.append(f"  {validation_cmds[cmd_idx]}")
        lines.append("  ```")
        lines.append("  Expected: This validation stage should pass ✅")
        lines.append("")

    return lines


def _build_hidden_problem_from_files(
    problem_num: int,
    additional_files: List[Dict[str, Any]],
    guidance_doc: Dict[str, Any],
    validation_cmds: List[str],
) -> List[str]:
    """Build hidden problem section from additional_files (no explicit linked_issue)."""

    lines = [
        f"## PROBLEM #{problem_num}: HIDDEN FILES (FIX AFTER PROBLEM #1)",
        "",
        "**Why These Files Are Hidden**:",
        "  - CI log only shows Problem #1 (first failure)",
        "  - These files are NOT mentioned in the CI error output",
        "  - But memory shows they need fixing too (based on similar past repairs)",
        "",
        f"**Files to Fix** ({len(additional_files)} files):",
    ]

    for af in additional_files[:20]:  # Limit to 20
        file_path = af.get("file", "")
        reason = af.get("reason", "")
        fix = af.get("fix", "")

        line = f"  - `{file_path}`"
        if reason:
            line += f" — {reason}"
        lines.append(line)

        if fix:
            lines.append(f"    → **Fix**: {fix}")

    if len(additional_files) > 20:
        lines.append(f"  - ... and {len(additional_files) - 20} more files")

    lines.append("")

    # Post-fix patterns
    post_fix_patterns = guidance_doc.get("post_fix_patterns", [])
    if post_fix_patterns:
        lines.append("**What Might Fail After Problem #1 is Fixed**:")
        for pattern in post_fix_patterns[:3]:
            likelihood = pattern.get("likelihood", "")
            pattern_text = pattern.get("pattern", "")
            how_to_fix = pattern.get("how_to_fix", "")

            lines.append(f"  - [{likelihood}] {pattern_text}")
            if how_to_fix:
                lines.append(f"    → {how_to_fix}")
        lines.append("")

    # Verification
    verification = guidance_doc.get("verification", {})
    verify_cmd = verification.get("command", "")
    if not verify_cmd and len(validation_cmds) > 1:
        verify_cmd = validation_cmds[1]

    if verify_cmd:
        lines.append("**Verification (REQUIRED)**:")
        lines.append("  ```bash")
        lines.append(f"  {verify_cmd}")
        lines.append("  ```")
        lines.append("")

    return lines
