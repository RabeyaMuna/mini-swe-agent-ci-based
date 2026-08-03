"""
Ablation Study Example - Measure Impact of Each Memory Level

Compares:
1. Baseline (no memory)
2. L1 only (repo+workflow specific)
3. L1+L2 (repo patterns)
4. L1+L2+L3 (full STAIR with universal patterns)

Usage:
    python ablation_experiment.py
"""

from memory_plugin import STAIRRetrieval
import openai
import json


def run_ablation_study():
    """
    Run ablation study to measure impact of each memory level.
    """

    # Initialize LLM client
    client = openai.OpenAI(api_key="your-api-key")

    # Test CI failure
    ci_failure = {
        'repo': 'CI-Repair/flower',
        'workflow': '.github/workflows/framework.yml',
        'problem_statement': 'Type checking failed with numpy.typing.DTypeLike annotation error',
        'error_signals': [
            'Cannot resolve type annotation "DTypeLike"',
            'mypy type checking failed'
        ],
        'config_signals': [
            'numpy.typing.mypy_plugin removed from pyproject.toml'
        ],
        'failure_type': 'type_checking'
    }

    # Initialize retrievers for each configuration
    print("Initializing retrievers...")

    baseline = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client,
        baseline_mode=True
    )

    l1_only = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client,
        memory_levels="l1"
    )

    l1_l2 = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client,
        memory_levels="l1+l2"
    )

    full_stair = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client,
        memory_levels="l1+l2+l3"
    )

    # Run retrieval for each configuration
    print("\n=== Running Ablation Study ===\n")

    configs = {
        'Baseline (no memory)': baseline,
        'L1 only': l1_only,
        'L1+L2': l1_l2,
        'L1+L2+L3 (Full STAIR)': full_stair
    }

    results = {}

    for config_name, retriever in configs.items():
        print(f"Running: {config_name}...")
        result = retriever.retrieve(ci_failure, top_k=5)
        results[config_name] = result

        # Print summary
        print(f"  Mode: {result['metadata']['mode']}")
        print(f"  Enabled levels: {result['metadata'].get('enabled_levels', [])}")
        print(f"  Retrieved: L1={result['metadata']['retrieved']['l1']}, "
              f"L2={result['metadata']['retrieved']['l2']}, "
              f"L3={result['metadata']['retrieved']['l3']}")
        print(f"  Common detected: {result['metadata']['common_detected']}")
        print(f"  Final problems: {result['metadata']['final']}")
        print()

    # Detailed comparison
    print("\n=== Detailed Comparison ===\n")

    for config_name, result in results.items():
        print(f"{config_name}:")
        print(f"  Total problems: {len(result['problems'])}")

        if result['problems']:
            print(f"  Problem types:")
            for problem in result['problems']:
                print(f"    - {problem.get('failure_type', 'unknown')}: {problem.get('problem', 'N/A')[:80]}...")
                print(f"      Confidence: {problem.get('confidence', 'N/A')}")
                print(f"      Source: {problem.get('source', {})}")
        else:
            print("  No problems found")

        print()

    # Analysis
    print("\n=== Analysis ===\n")

    baseline_count = len(results['Baseline (no memory)']['problems'])
    l1_count = len(results['L1 only']['problems'])
    l1_l2_count = len(results['L1+L2']['problems'])
    full_count = len(results['L1+L2+L3 (Full STAIR)']['problems'])

    print(f"Impact of L1: +{l1_count - baseline_count} problems (vs baseline)")
    print(f"Impact of L2: +{l1_l2_count - l1_count} problems (vs L1 only)")
    print(f"Impact of L3: +{full_count - l1_l2_count} problems (vs L1+L2)")
    print(f"\nTotal improvement: {full_count - baseline_count} problems (Full STAIR vs Baseline)")

    # Save results
    with open('ablation_results.json', 'w') as f:
        # Convert to JSON-serializable format
        serializable_results = {}
        for config_name, result in results.items():
            serializable_results[config_name] = {
                'metadata': result['metadata'],
                'problem_count': len(result['problems']),
                'problems': result['problems'][:3]  # First 3 for brevity
            }
        json.dump(serializable_results, f, indent=2)

    print("\nResults saved to ablation_results.json")


def compare_problem_quality():
    """
    Compare quality of problems from different levels.

    Measures:
    - Root cause completeness
    - Repair action detail
    - Signal evidence
    - Cross-level synthesis
    """

    client = openai.OpenAI(api_key="your-api-key")

    ci_failure = {
        'repo': 'CI-Repair/flower',
        'workflow': '.github/workflows/framework.yml',
        'problem_statement': 'Dependency resolution failed',
        'error_signals': ['Could not find matching version of docstrfmt'],
        'config_signals': ["docstrfmt = '==2.0.2' in pyproject.toml"],
        'failure_type': 'dependency_version'
    }

    # Compare L1 vs L1+L2 vs Full
    l1_only = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client,
        memory_levels="l1"
    )

    full = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client,
        memory_levels="l1+l2+l3"
    )

    print("\n=== Quality Comparison ===\n")

    l1_result = l1_only.retrieve(ci_failure, top_k=5)
    full_result = full.retrieve(ci_failure, top_k=5)

    if l1_result['problems']:
        l1_problem = l1_result['problems'][0]
        print("L1 Only - First Problem:")
        print(f"  Root Cause: {l1_problem.get('root_cause', 'N/A')}")
        print(f"  Rationale: {l1_problem.get('rationale', {})}")
        print(f"  Signals: {len(l1_problem.get('signals', {}).get('error_signals', []))} error signals")
        print(f"  Sources: {l1_problem.get('source', {})}")

    print()

    if full_result['problems']:
        full_problem = full_result['problems'][0]
        print("Full STAIR (L1+L2+L3) - First Problem:")
        print(f"  Root Cause: {full_problem.get('root_cause', 'N/A')}")
        print(f"  Rationale: {full_problem.get('rationale', {})}")
        print(f"  Signals: {len(full_problem.get('signals', {}).get('error_signals', []))} error signals")
        print(f"  Sources: {full_problem.get('source', {})}")

    print("\nOK Full STAIR should show:")
    print("  - More complete root cause (synthesized from L1+L2+L3)")
    print("  - WHY/HOW/WHAT rationale breakdown")
    print("  - Signals from multiple levels")
    print("  - Sources from L1, L2, AND L3")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'quality':
        compare_problem_quality()
    else:
        run_ablation_study()
