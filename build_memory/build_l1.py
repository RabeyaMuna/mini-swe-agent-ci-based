"""
L1 Memory Builder - Problem-Level Concrete Failures.

This module contains ALL logic that was previously in L2:
1. Dependency analysis (caller-callee, problem ordering)
2. Prompt generation (repair sequence prompts)
3. LLM invocation (to create repair sequences)
4. Problem organization with "enabled" relationships

Each L1 entry represents ONE CI issue with multiple problems:
- Problem description and root cause
- Fix strategy (how and why it works)
- Affected files
- Verification command
- Problem dependencies (which problems enable others)

This is the NEW L1 - migrated from the old L2 repair sequence structure.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import L1 prompts (repair sequence generation)
try:
    from prompt_template.memory_build import (
        build_l1_full_sequence_prompt,
        build_l1_validation_group_prompt,
    )

    _PROMPTS_AVAILABLE = True
except ImportError:
    logger.warning("[L1 Build] L1 prompt templates not available")
    _PROMPTS_AVAILABLE = False

# Import dependency utilities
try:
    from utilities.dependency_evidence import (
        build_dependency_graph_from_structured_diff,
    )

    _DEPENDENCY_AVAILABLE = True
except ImportError:
    logger.warning("[L1 Build] Dependency utilities not available")
    _DEPENDENCY_AVAILABLE = False


def generate_l1_from_decomposed_problems(
    *,
    issue_id: str,
    repo: str,
    repo_owner: str,
    workflow_path: str,
    decomposed_problems: List[Dict[str, Any]],
    dependencies: Dict[str, Any],
    ground_truth_files: List[str],
    llm: Any = None,  # Kept for backward compatibility but not used
    repo_name: str = None,  # Optional, will be extracted from repo if not provided
) -> Dict[str, Any]:
    """
    Generate L1 memory from decomposed problems.

    L1 is JUST DATA FORMATTING - no LLM needed!

    This function:
    1. Takes decomposed problems (from decomposition stage)
    2. Adds "enabled" relationships from dependencies
    3. Formats into L1 structure
    4. Returns structured L1 memory

    Note: LLM generation happens in L2 (build_l2_memory), not here.

    Args:
        issue_id: CI issue ID
        repo: Repository name (owner/repo)
        repo_owner: Repository owner
        workflow_path: Path to workflow file
        decomposed_problems: List of problems from decomposition
        dependencies: Dependency graph with enabled relationships
        ground_truth_files: List of files changed in the fix
        llm: (Deprecated - kept for backward compatibility, not used)
        repo_name: Repository name (just the repo part, not owner/repo)

    Returns:
        L1 memory dictionary with problems and enabled relationships
    """
    # Extract repo_name from repo if not provided
    if repo_name is None:
        repo_name = repo.split("/")[1] if "/" in repo else repo

    if not decomposed_problems:
        logger.warning(f"[L1 Build] No decomposed problems for {issue_id}")
        return _empty_l1_entry(
            issue_id, repo, repo_owner, repo_name, workflow_path, ground_truth_files
        )

    # L1 uses LLM to build REPAIR SEQUENCE with ordered problems + enabled relationships
    # This creates a sequential repair order, not just a flat list

    if llm is None:
        logger.warning("[L1 Build] No LLM provided, using mechanical conversion")
        # Fallback: just add enabled relationships without LLM
        problems_with_enabled = _add_enabled_relationships(
            problems=decomposed_problems,
            dependencies=dependencies,
        )
        repair_sequence = {"problems": problems_with_enabled}
    else:
        # Use LLM to generate repair sequence with proper ordering
        repair_sequence = _generate_repair_sequence_llm(
            problems=decomposed_problems,
            dependencies=dependencies,
            llm=llm,
            issue_id=issue_id,
            repo=repo,
        )

    # Extract problems from repair sequence
    problems = repair_sequence.get("problems", [])

    if not problems:
        logger.warning(f"[L1 Build] LLM generated no problems for {issue_id}")
        # Fallback to mechanical conversion
        problems = _mechanical_problem_conversion(decomposed_problems)

    # ALWAYS add/ensure enabled relationships (even if LLM added them)
    # This ensures consistency and correctness based on dependency graph
    problems = _add_enabled_relationships(
        problems=problems,
        dependencies=dependencies,
    )

    # Build L1 entry with enabled relationships
    l1_entry = _build_l1_entry_with_dependencies(
        issue_id=issue_id,
        repo=repo,
        repo_owner=repo_owner,
        repo_name=repo_name,
        workflow_path=workflow_path,
        problems=problems,
        ground_truth_files=ground_truth_files,
    )

    logger.info(
        f"[L1 Build] Generated L1 memory for {issue_id}: "
        f"{len(l1_entry['problems'])} problems"
    )

    return l1_entry


def build_l1_memory(
    *,
    issue_id: str,
    repo: str,
    repo_owner: str,
    workflow_path: str,
    l2_repair_trajectory: Dict[str, Any],
    ground_truth_files: List[str],
) -> Dict[str, Any]:
    """
    Build L1 memory from EXISTING L2 repair trajectory.

    Use this when you already have a repair sequence (e.g., from saved data).
    For generating NEW L1 from decomposed problems, use generate_l1_from_decomposed_problems().

    Args:
        issue_id: CI issue ID
        repo: Repository name (owner/repo)
        repo_owner: Repository owner
        workflow_path: Path to workflow file
        l2_repair_trajectory: EXISTING L2 repair sequence with problems array
        ground_truth_files: List of files changed in the fix

    Returns:
        L1 memory dictionary ready to be saved to failure_memory.json
    """
    problems = l2_repair_trajectory.get("problems", [])

    if not problems:
        logger.warning(f"[L1 Build] No problems found in L2 trajectory for {issue_id}")
        return _empty_l1_entry(
            issue_id, repo, repo_owner, workflow_path, ground_truth_files
        )

    # Build L1 entry with enabled relationships
    l1_entry = _build_l1_entry_with_dependencies(
        issue_id=issue_id,
        repo=repo,
        repo_owner=repo_owner,
        repo_name=repo_name,
        workflow_path=workflow_path,
        problems=problems,
        ground_truth_files=ground_truth_files,
    )

    logger.info(
        f"[L1 Build] Built L1 memory for {issue_id}: "
        f"{len(l1_entry['problems'])} problems extracted"
    )

    return l1_entry


def normalize_l1_record(l1_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize L1 record for storage.

    Ensures consistent field types and formats:
    - Files are lists of strings
    - Text fields are stripped
    - Problem IDs are integers
    """
    normalized = dict(l1_record)

    # Normalize problems array
    if "problems" in normalized:
        normalized_problems = []
        for problem in normalized["problems"]:
            if not isinstance(problem, dict):
                continue

            norm_problem = {
                "problem_id": int(problem.get("problem_id", 0))
                if problem.get("problem_id")
                else None,
                "verification_cmd": str(problem.get("verification_cmd", "")).strip(),
                "failure_type": str(problem.get("failure_type", "")).strip(),
                "problem": str(problem.get("problem", "")).strip(),
                "root_cause": str(problem.get("root_cause", "")).strip(),
                "fix_strategy": str(problem.get("fix_strategy", "")).strip(),
                "files": _normalize_file_list(problem.get("files", [])),
            }

            # Add optional fields
            if "enabled" in problem:
                norm_problem["enabled"] = problem["enabled"]

            normalized_problems.append(norm_problem)

        normalized["problems"] = normalized_problems

    # Normalize changed_files
    if "changed_files" in normalized:
        normalized["changed_files"] = _normalize_file_list(normalized["changed_files"])

    return normalized


def _normalize_file_list(files: Any) -> List[str]:
    """Normalize file list to consistent format."""
    if not files:
        return []
    if isinstance(files, str):
        return [files.strip()]
    if isinstance(files, list):
        return [str(f).strip() for f in files if f]
    return []


def _add_enabled_relationships(
    problems: List[Dict[str, Any]],
    dependencies: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Add 'enabled' relationships from dependencies to problems.

    This maps the dependency graph to the enabled field in each problem,
    showing which problems are revealed after fixing this one.

    Args:
        problems: List of decomposed problems
        dependencies: Dependency dict with 'dependency_edges' and 'repair_order'

    Returns:
        List of problems with 'enabled' field added
    """
    # Create a mapping from problem to its enabled problems
    enabled_map = {}

    # Extract dependency edges (format: [[from_id, to_id], ...] or [{"from": id, "to": id}, ...])
    edges = dependencies.get("dependency_edges", [])

    # Debug: Check edge format
    if edges and len(edges) > 0:
        print(f"  [DEBUG] First edge type: {type(edges[0])}, sample: {edges[0]}")

    for edge in edges:
        # Handle both list/tuple format and dict format
        from_id = None
        to_id = None

        if isinstance(edge, dict):
            # Dict format: try various key names
            from_id = edge.get("from") or edge.get("from_id") or edge.get("caller") or edge.get("source")
            to_id = edge.get("to") or edge.get("to_id") or edge.get("callee") or edge.get("target")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            # List/tuple format
            from_id, to_id = edge[0], edge[1]
        else:
            # Unknown format - skip
            print(f"  [DEBUG] Skipping invalid edge format: {type(edge)} - {edge}")
            continue

        if from_id and to_id:
            if from_id not in enabled_map:
                enabled_map[from_id] = []
            enabled_map[from_id].append(to_id)
        else:
            print(f"  [DEBUG] Skipping edge with missing IDs: from={from_id}, to={to_id}, edge={edge}")

    # Add enabled field to each problem
    problems_with_enabled = []
    for idx, problem in enumerate(problems, 1):
        problem_copy = dict(problem)

        # Add problem_id if not present
        if "problem_id" not in problem_copy:
            problem_copy["problem_id"] = idx

        # ALWAYS add enabled field (empty array if no dependencies)
        problem_id = problem_copy.get("problem_id", idx)
        problem_copy["enabled"] = enabled_map.get(problem_id, [])

        problems_with_enabled.append(problem_copy)

    return problems_with_enabled


def _empty_l1_entry(
    issue_id: str,
    repo: str,
    repo_owner: str,
    repo_name: str,
    workflow_path: str,
    ground_truth_files: List[str],
) -> Dict[str, Any]:
    """Create empty L1 entry when no problems available."""
    workflow_name = workflow_path.split("/")[-1] if workflow_path else ""
    return {
        "issue_id": issue_id,
        "repo": repo,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "workflow": workflow_path,
        "workflow_name": workflow_name,
        "changed_files": ground_truth_files,
        "problems": [],
    }


def _reorder_problems_by_ci_sequence(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reorder problems by CI verification sequence and dependencies.

    Order:
    1. validation_order (CI pipeline: dependencies → build → type check → test → docs)
    2. Config files first (pyproject.toml, setup.py, etc.)
    3. Dependencies first (has dependency_type)
    4. Non-cascading before cascading
    5. problem_id (original order as tiebreaker)

    This ensures problems are presented in the order they would fail in CI.
    """
    def is_config_file(filepath: str) -> bool:
        """Check if file is a configuration file."""
        config_patterns = [
            "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
            "package.json", "tsconfig.json", ".pre-commit-config",
            "Dockerfile", ".github/workflows",
        ]
        return any(pattern in filepath for pattern in config_patterns)

    def problem_sort_key(problem: dict) -> tuple:
        """Generate sort key for CI verification order."""
        validation_order = problem.get("validation_order", 999)

        # Config file bonus (comes first)
        files = problem.get("files", []) or problem.get("affected_files", [])
        is_config = any(is_config_file(f) for f in files)
        config_rank = 0 if is_config else 1

        # Dependency rank
        has_dependency = bool(problem.get("dependency_type", ""))
        dependency_rank = 0 if has_dependency else 1

        # Cascading rank (non-cascading first)
        is_cascading = problem.get("is_cascading", False)
        cascading_rank = 1 if is_cascading else 0

        # Original problem_id
        problem_id = problem.get("problem_id", 0)

        return (validation_order, config_rank, dependency_rank, cascading_rank, problem_id)

    # Sort problems
    sorted_problems = sorted(problems, key=problem_sort_key)

    # Reassign problem_ids to match new order
    for idx, problem in enumerate(sorted_problems, 1):
        problem["problem_id"] = idx

    return sorted_problems


def _build_l1_entry_with_dependencies(
    *,
    issue_id: str,
    repo: str,
    repo_owner: str,
    repo_name: str,
    workflow_path: str,
    problems: List[Dict[str, Any]],
    ground_truth_files: List[str],
) -> Dict[str, Any]:
    """
    Build L1 entry structure with problem dependencies (enabled relationships).

    Problems are automatically reordered by CI verification sequence before saving.
    """
    # REORDER PROBLEMS BY CI VERIFICATION SEQUENCE
    problems = _reorder_problems_by_ci_sequence(problems)

    # Extract workflow name from path
    workflow_name = workflow_path.split("/")[-1] if workflow_path else ""

    # Build L1 entry structure
    l1_entry = {
        "issue_id": issue_id,
        "repo": repo,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "workflow": workflow_path,
        "workflow_name": workflow_name,
        "problems": [],
    }

    # Process each problem from repair sequence (now in CI order!)
    for problem in problems:
        if not isinstance(problem, dict):
            continue

        # Extract core problem fields (remove estimated_time and pattern_detected)
        l1_problem = {
            "problem_id": problem.get("problem_id"),
            "verification_cmd": problem.get("verification_cmd", ""),
            "failure_type": problem.get("failure_type", ""),
            "problem": problem.get("problem", ""),
            "root_cause": problem.get("root_cause", ""),
            "fix_strategy": problem.get("fix_strategy", ""),
            "files": problem.get("files", []),
            "enabled": problem.get("enabled", []),  # KEEP enabled field!
        }

        # Add optional "enabled" field (problem dependencies)
        # This shows which problems are revealed/enabled after fixing this one
        if "enabled" in problem and problem["enabled"]:
            l1_problem["enabled"] = problem["enabled"]

        # Validate that problem has essential fields
        if not l1_problem["problem"] or not l1_problem["root_cause"]:
            logger.warning(
                f"[L1 Build] Skipping problem {l1_problem['problem_id']} - "
                f"missing essential fields (problem or root_cause)"
            )
            continue

        l1_entry["problems"].append(l1_problem)

    return l1_entry


def _generate_repair_sequence_llm(
    *,
    problems: List[Dict[str, Any]],
    dependencies: Dict[str, Any],
    llm: Any,
    issue_id: str,
    repo: str,
) -> Dict[str, Any]:
    """
    Generate repair sequence using LLM with Three-Tier Strategy.

    Tier 1 (<10 problems): Single-pass LLM (best quality)
    Tier 2 (10-20 problems): Grouped LLM per validation (good quality)
    Tier 3 (>20 problems): Mechanical generation (guaranteed complete)
    """
    if not llm or not _PROMPTS_AVAILABLE:
        logger.warning(
            "[L1 Build] LLM or prompts not available, using mechanical conversion"
        )
        return {"problems": _mechanical_problem_conversion(problems)}

    actual_count = len(problems)
    prompt_size = len(json.dumps(problems, indent=2))

    logger.info(f"[L1 Build] L2 Strategy: {actual_count} problems, {prompt_size} chars")

    # TIER 1: Small dataset - Single-pass LLM
    if actual_count <= 10 and prompt_size < 20000:
        logger.info("[L1 Build] -> Tier 1: Single-pass LLM")
        try:
            result = _l2_tier1_single_pass(problems, dependencies, llm, issue_id, repo)
            if len(result.get("problems", [])) >= actual_count * 0.9:
                logger.info(
                    f"[L1 Build] Tier 1 success: {len(result['problems'])} problems"
                )
                return result
            else:
                logger.warning(
                    f"[L1 Build] Tier 1 incomplete ({len(result.get('problems', []))}/{actual_count})"
                )
        except Exception as e:
            logger.warning(f"[L1 Build] Tier 1 failed: {str(e)[:100]}")

    # TIER 2: Medium dataset - Grouped LLM
    if actual_count <= 20:
        logger.info("[L1 Build] -> Tier 2: Grouped LLM per validation")
        try:
            result = _l2_tier2_grouped_llm(problems, dependencies, llm, issue_id, repo)
            if len(result.get("problems", [])) >= actual_count * 0.9:
                logger.info(
                    f"[L1 Build] Tier 2 success: {len(result['problems'])} problems"
                )
                return result
            else:
                logger.warning(
                    f"[L1 Build] Tier 2 incomplete ({len(result.get('problems', []))}/{actual_count})"
                )
        except Exception as e:
            logger.warning(f"[L1 Build] Tier 2 failed: {str(e)[:100]}")

    # TIER 3: Large dataset or fallback - Mechanical
    logger.info("[L1 Build] -> Tier 3: Mechanical generation (guaranteed complete)")
    mechanical_problems = _mechanical_problem_conversion(problems)
    logger.info(f"[L1 Build] Tier 3 success: {len(mechanical_problems)} problems")
    return {"problems": mechanical_problems}


def _l2_tier1_single_pass(
    problems: List[Dict], dependencies: Dict, llm: Any, issue_id: str, repo: str
) -> Dict[str, Any]:
    """Tier 1: Single-pass LLM for small datasets."""
    if not _PROMPTS_AVAILABLE:
        raise ImportError("L2 prompt templates not available")

    # Prepare data for LLM
    problems_for_llm = []
    for idx, prob in enumerate(problems, 1):
        problems_for_llm.append(
            {
                "id": idx,
                "validation_order": prob.get("validation_order", 999),
                "validation_cmd": prob.get("validation_cmd", ""),
                "problem_type": prob.get("problem_type", ""),
                "what_broke": prob.get("what_broke", ""),
                "root_cause": prob.get("root_cause", ""),
                "how_fixed": prob.get("how_fixed", ""),
                "why_fixed_works": prob.get("why_fixed_works", ""),
                "files": prob.get("affected_files", []) or prob.get("files", []),
                "issue_type": prob.get("issue_type", ""),
            }
        )

    # Use L1 prompt template
    STRICT_JSON_RULES = """
CRITICAL JSON RULES:
1. Return ONLY valid JSON - no markdown, no code fences, no explanations
2. All strings must use double quotes "
3. No trailing commas in arrays or objects
4. Escape special characters in strings (\\n, \\", \\\\)
5. Arrays and objects must be properly closed
"""

    prompt = build_l1_full_sequence_prompt(
        issue_id=issue_id,
        repo=repo,
        problems=problems_for_llm,
        dependencies=dependencies,
        strict_json_rules=STRICT_JSON_RULES,
    )

    time.sleep(3)  # Rate limiting
    response = _invoke_json(llm, prompt)

    if isinstance(response, dict) and "problems" in response:
        return response
    else:
        raise ValueError("Invalid response from LLM")


def _l2_tier2_grouped_llm(
    problems: List[Dict], dependencies: Dict, llm: Any, issue_id: str, repo: str
) -> Dict[str, Any]:
    """Tier 2: Grouped LLM per validation for medium datasets."""
    if not _PROMPTS_AVAILABLE:
        raise ImportError("L2 prompt templates not available")

    # Group by validation_cmd
    validation_groups = defaultdict(list)
    for prob in problems:
        cmd = prob.get("validation_cmd", "unknown")
        validation_groups[cmd].append(prob)

    logger.info(f"[L1 Build] Grouped into {len(validation_groups)} validations")

    all_l2_problems = []

    STRICT_JSON_RULES = """
CRITICAL JSON RULES:
1. Return ONLY valid JSON - no markdown, no code fences, no explanations
2. All strings must use double quotes "
3. No trailing commas in arrays or objects
"""

    for validation_cmd, group_problems in validation_groups.items():
        logger.info(
            f"[L1 Build] Processing {validation_cmd}: {len(group_problems)} problems"
        )

        # Prepare problems for this validation
        problems_for_llm = []
        for idx, prob in enumerate(group_problems, 1):
            problems_for_llm.append(
                {
                    "id": idx,
                    "validation_order": prob.get("validation_order", 999),
                    "validation_cmd": prob.get("validation_cmd", ""),
                    "problem_type": prob.get("problem_type", ""),
                    "what_broke": prob.get("what_broke", ""),
                    "root_cause": prob.get("root_cause", ""),
                    "how_fixed": prob.get("how_fixed", ""),
                    "why_fixed_works": prob.get("why_fixed_works", ""),
                    "files": prob.get("affected_files", []) or prob.get("files", []),
                    "issue_type": prob.get("issue_type", ""),
                }
            )

        prompt = build_l1_validation_group_prompt(
            validation_cmd=validation_cmd,
            problems=problems_for_llm,
            strict_json_rules=STRICT_JSON_RULES,
        )

        time.sleep(2)
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict) and "problems" in response:
            # Add metadata back to problems
            for l2_prob in response["problems"]:
                orig_idx = l2_prob.get("problem_id", 1) - 1
                if 0 <= orig_idx < len(group_problems):
                    orig_prob = group_problems[orig_idx]
                    l2_prob["problem_type"] = orig_prob.get("problem_type")
                    l2_prob["validation_order"] = orig_prob.get("validation_order", 999)

            all_l2_problems.extend(response["problems"])
        else:
            # Fallback: use mechanical for this validation
            all_l2_problems.extend(_mechanical_problem_conversion(group_problems))

    return {"problems": all_l2_problems}


def _mechanical_problem_conversion(
    problems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Mechanical conversion of decomposed problems to L1 format.

    Used as fallback when LLM is not available or fails.
    """
    converted = []
    for idx, prob in enumerate(problems, 1):
        converted.append(
            {
                "problem_id": idx,
                "verification_cmd": prob.get("validation_cmd", ""),
                "failure_type": prob.get("failure_type", prob.get("problem_type", "")),
                "problem": prob.get("what_broke", prob.get("problem", "")),
                "root_cause": prob.get("root_cause", ""),
                "fix_strategy": _merge_fix_strategy(
                    prob.get("how_fixed", ""), prob.get("why_fixed_works", "")
                ),
                "files": prob.get("affected_files", []) or prob.get("files", []),
            }
        )
    return converted


def _merge_fix_strategy(how_fixed: str, why_works: str) -> str:
    """Merge how_fixed and why_fixed_works into single fix_strategy."""
    parts = []
    if how_fixed:
        parts.append(how_fixed.strip())
    if why_works:
        parts.append(why_works.strip())
    return " ".join(parts)


def _invoke_json(llm: Any, prompt: str) -> Dict[str, Any]:
    """Invoke LLM and parse JSON response."""
    try:
        # Try LangChain interface
        from langchain_core.messages import HumanMessage

        result = llm.invoke([HumanMessage(content=prompt)])
        response_text = getattr(result, "content", None) or str(result)
    except (ImportError, AttributeError):
        # Try direct call
        result = llm.invoke(prompt)
        response_text = getattr(result, "content", None) or str(result)

    # Parse JSON from response
    response_text = response_text.strip()

    # Remove markdown fences
    if "```json" in response_text:
        response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()

    return json.loads(response_text)


def create_search_document_l1(l1_record: Dict[str, Any]) -> str:
    """
    Create search document for L1 record embedding.

    Concatenates all problem information for semantic search:
    - Verification commands (what validators were run)
    - Failure types (categories of errors)
    - Problem descriptions
    - Root causes
    - Fix strategies
    - File paths
    """
    parts = []

    # Add repo context
    if l1_record.get("repo"):
        parts.append(f"repo:{l1_record['repo']}")

    if l1_record.get("workflow"):
        parts.append(f"workflow:{l1_record['workflow']}")

    # Add problem details
    for problem in l1_record.get("problems", []):
        if not isinstance(problem, dict):
            continue

        # Verification command
        if problem.get("verification_cmd"):
            parts.append(f"cmd:{problem['verification_cmd']}")

        # Failure type
        if problem.get("failure_type"):
            parts.append(f"type:{problem['failure_type']}")

        # Problem description
        if problem.get("problem"):
            parts.append(f"problem:{problem['problem']}")

        # Root cause
        if problem.get("root_cause"):
            parts.append(f"cause:{problem['root_cause']}")

        # Fix strategy
        if problem.get("fix_strategy"):
            parts.append(f"fix:{problem['fix_strategy']}")

        # Files
        files = problem.get("files", [])
        if files:
            parts.append(f"files:{' '.join(files)}")

    return " | ".join(parts)
