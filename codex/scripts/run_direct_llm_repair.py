#!/usr/bin/env python3
"""
Run CI repair using YOUR OWN LLM (GLM, Minimax, etc.) instead of Codex CLI.

This uses the same prompt preparation as run_codex_ci_repair.py but sends
the prompt to your LLM via utilities/llm_model.py.

Usage:
    # With GLM 5.2
    python3 codex/scripts/run_direct_llm_repair.py \
        --issue-ids 125 \
        --ablations baseline \
        --model glm5.2

    # With Minimax
    python3 codex/scripts/run_direct_llm_repair.py \
        --issue-ids 125 \
        --ablations L1+L2+L3 \
        --model minimax \
        --memory-root data/back_trs

    # Multiple issues, all ablations
    python3 codex/scripts/run_direct_llm_repair.py \
        --issue-ids 125,126,127 \
        --ablations baseline,L1,L1+L2,L1+L2+L3 \
        --model glm5.2 \
        --memory-root data/back_trs
"""

# Import everything from the original script
import sys
from pathlib import Path

# Add parent directory to path to import from run_codex_ci_repair
sys.path.insert(0, str(Path(__file__).parent))

# Import all the functions we need from the original script
from run_codex_ci_repair import (
    parse_args,
    load_issue_dataset,
    load_ci_failure_analysis,
    load_workflow_validation,
    retrieve_memory_for_issue,
    build_unified_problem_list,
    compose_issue_document,
    write_issue_document,
    git_diff,
    changed_files,
    write_text,
    DEFAULT_OUTPUT_ROOT,
)

# Import your LLM
sys.path.insert(0, str(Path(__file__).parents[2]))
from utilities.llm_model import LitellmModel

def run_llm_agent(
    checkout: Path,
    prompt: str,
    model_name: str,
    transcript_path: Path,
    timeout: int = 3600,
) -> dict:
    """
    Run LLM agent on the prompt using your own LLM.

    Args:
        checkout: Repository checkout path
        prompt: The repair task prompt
        model_name: Model to use (glm5.2, minimax, etc.)
        transcript_path: Where to save the transcript
        timeout: Timeout in seconds

    Returns:
        Dict with returncode, elapsed_seconds
    """
    import time

    started = time.time()

    print(f"\n{'='*80}")
    print(f"[LLM AGENT] Starting with model: {model_name}")
    print(f"[LLM AGENT] Working directory: {checkout}")
    print(f"[LLM AGENT] Output will be saved to: {transcript_path}")
    print(f"{'='*80}\n")

    # Create LLM
    llm = LitellmModel(model_name)

    # Stream output to both terminal and file
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(transcript_path, 'w', encoding='utf-8') as f:
            # Write prompt
            f.write("=" * 80 + "\n")
            f.write("PROMPT\n")
            f.write("=" * 80 + "\n")
            f.write(prompt + "\n")
            f.write("=" * 80 + "\n")
            f.write("RESPONSE\n")
            f.write("=" * 80 + "\n")

            # Invoke LLM
            print(f"Sending prompt to {model_name}...")
            result = llm.invoke(prompt)

            # Get response
            response = result.content

            # Write and print response
            print("\n" + "=" * 80)
            print("LLM RESPONSE:")
            print("=" * 80)
            print(response)
            print("=" * 80 + "\n")

            f.write(response + "\n")

        elapsed = time.time() - started

        print(f"\n{'='*80}")
        print(f"[LLM AGENT] Completed in {elapsed:.1f}s")
        print(f"{'='*80}\n")

        return {"returncode": 0, "elapsed_seconds": elapsed}

    except Exception as e:
        elapsed = time.time() - started
        print(f"\n{'='*80}")
        print(f"[LLM AGENT] ERROR: {e}")
        print(f"{'='*80}\n")

        with open(transcript_path, 'a', encoding='utf-8') as f:
            f.write(f"\nERROR: {e}\n")

        return {"returncode": 1, "elapsed_seconds": elapsed}


def run_issue(args, ablation: str, issue: dict):
    """Run repair on a single issue - modified to use LLM instead of Codex CLI."""
    import json

    # Import needed functions
    from run_codex_ci_repair import (
        repo_slug,
        prediction_model_name,
        DEFAULT_LOG_CACHE,
        DEFAULT_WORKFLOW_CACHE,
        DEFAULT_MEMORY_ROOT,
        clone_repository,
        extract_problem_list,
    )

    issue_id = issue["id"]
    result_dir = (
        DEFAULT_OUTPUT_ROOT
        / "direct_llm"
        / f"{ablation.replace('+', '_').lower()}_{args.model.replace('.', '_')}"
        / str(issue_id)
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[codex-ci-repair] issue={issue_id} ablation={ablation} model={args.model}")

    # Clone repository
    from run_codex_ci_repair import clone_repository
    checkout = clone_repository(issue, result_dir)

    # Load CI failure analysis
    ci_failure = load_ci_failure_analysis(
        issue,
        DEFAULT_LOG_CACHE,
        args.generate_missing_analysis,
        args.context_model or args.model,
    )
    write_text(result_dir / "ci_failure.json", json.dumps(ci_failure, indent=2))

    # Load workflow validation
    verification = load_workflow_validation(
        issue,
        DEFAULT_WORKFLOW_CACHE,
        args.generate_missing_analysis,
        args.context_model or args.model,
    )

    # Retrieve memory
    memory_retrieval, memory_md = retrieve_memory_for_issue(
        issue,
        ci_failure,
        result_dir,
        args.memory_root or DEFAULT_MEMORY_ROOT,
        ablation,
        args.memory_top_k,
        args.context_model or args.model,
    )

    # Build unified problem list
    ci_problems = extract_problem_list(ci_failure)
    memory_problems = memory_retrieval.get("problems", [])

    problems = build_unified_problem_list(
        ci_failure=ci_failure,
        ci_problems=ci_problems,
        memory_problems=memory_problems,
        ablation=ablation,
    )

    print(f"\n[DEBUG] Total problems: {len(problems)}")
    for idx, p in enumerate(problems, 1):
        has_repair = '✓' if p.get('repair_strategy') else '✗'
        print(f"[DEBUG]   {idx}. [repair:{has_repair}] {p.get('problem', 'N/A')[:70]}...")
    print()

    # Process each problem
    problem_results = []
    for problem in problems:
        document = compose_issue_document(
            issue,
            ci_failure,
            verification,
            problem,
            len(problems),
            ablation,
        )

        prompt_path = write_issue_document(result_dir, problem, document)
        transcript_path = result_dir / f"llm_transcript_problem_{problem['number']}.txt"

        # Run with LLM instead of Codex CLI
        run_result = run_llm_agent(
            checkout,
            document,
            args.model,
            transcript_path,
            args.timeout,
        )

        # Collect results
        problem_results.append({
            "problem_number": problem['number'],
            "returncode": run_result["returncode"],
            "elapsed_seconds": run_result["elapsed_seconds"],
        })

        if run_result["returncode"] != 0:
            print(f"Problem {problem['number']} failed, skipping remaining problems")
            break

    # Save prediction
    prediction = {
        "model_name_or_path": args.model,
        "instance_id": str(issue_id),
        "model_patch": git_diff(checkout),
        "changed_files": changed_files(checkout),
        "problem_results": problem_results,
    }

    write_text(
        result_dir / "prediction.json",
        json.dumps(prediction, indent=2)
    )

    print(f"✓ Completed issue {issue_id}")


def main():
    """Main entry point."""
    # Parse args (reuse from original script)
    args = parse_args()

    # Add model argument if not present
    if not hasattr(args, 'model') or not args.model:
        print("ERROR: --model is required!")
        print("Example: --model glm5.2")
        sys.exit(1)

    # Load issues
    issues = load_issue_dataset(args)
    ablations = [a.strip() for a in args.ablations.split(",")]

    print(f"[codex-ci-repair] Issues to process: {len(issues)}")
    print(f"[codex-ci-repair] Ablations: {', '.join(ablations)}")
    print(f"[codex-ci-repair] Model: {args.model}")

    # Run each ablation
    for ablation in ablations:
        print(f"\n[codex-ci-repair] Starting ablation: {ablation}")

        for issue in issues:
            try:
                run_issue(args, ablation, issue)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                sys.exit(1)
            except Exception as e:
                print(f"ERROR processing issue {issue.get('id')}: {e}")
                import traceback
                traceback.print_exc()

        print(f"[codex-ci-repair] Completed ablation: {ablation}")


if __name__ == "__main__":
    main()
