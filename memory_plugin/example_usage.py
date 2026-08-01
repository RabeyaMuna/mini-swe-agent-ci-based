"""
Example usage of STAIR Memory Retrieval System
"""

from memory_plugin import STAIRRetrieval


def example_with_openai():
    """Example using OpenAI client."""
    import openai

    client = openai.OpenAI(api_key="your-api-key")

    retrieval = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client
    )

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

    result = retrieval.retrieve(ci_failure, top_k=5)

    print("=== RETRIEVAL RESULTS ===\n")
    print(f"Retrieved: {result['metadata']['retrieved']}")
    print(f"Common detected: {result['metadata']['common_detected']}")
    print(f"Filtered: {result['metadata']['filtered']}")
    print(f"Final problems: {result['metadata']['final']}")

    print("\n=== PROBLEMS TO FIX ===\n")
    for i, problem in enumerate(result['problems'], 1):
        print(f"{i}. {problem['problem']}")
        print(f"   Root Cause: {problem['root_cause']}")
        print(f"   Type: {problem['type']}")
        print(f"   Confidence: {problem['confidence']}")
        print(f"   Steps: {len(problem['repair_actions']['steps'])}")
        print()


def example_with_anthropic():
    """Example using Anthropic client."""
    import anthropic

    client = anthropic.Anthropic(api_key="your-api-key")

    retrieval = STAIRRetrieval(
        memory_dir='../data/fwr_trs',
        llm_client=client
    )

    ci_failure = {
        'repo': 'CI-Repair/flower',
        'workflow': '.github/workflows/framework.yml',
        'problem_statement': 'Poetry install failed - dependency resolution error',
        'error_signals': [
            'Could not find a matching version of package docstrfmt',
            'Dependency resolution failed'
        ],
        'config_signals': [
            "docstrfmt = '==2.0.2' in pyproject.toml"
        ],
        'failure_type': 'dependency_version'
    }

    result = retrieval.retrieve(ci_failure, top_k=5)

    for problem in result['problems']:
        print(f"Problem: {problem['problem']}")
        print(f"Files: {problem['repair_actions']['files']}")
        print(f"Validation: {problem['repair_actions']['validation_cmd']}")
        print()


if __name__ == '__main__':
    # Choose one
    example_with_openai()
    # example_with_anthropic()
