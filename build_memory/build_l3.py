"""
L3 universal pattern builder.

Generates reusable cross-repo patterns from L1 + L2 using LLM.
"""

from typing import Any, Dict
import json

# Import L3 prompt from centralized location
from prompt_template.memory_build import build_l3_prompt


def extract_text(response) -> str:
    """Extract text from LLM response."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is None:
        return getattr(response, "text", "") or str(response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
        return "".join(parts)
    return str(content)


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling code fences and control characters."""
    import re

    if not text or not str(text).strip():
        raise ValueError("Empty LLM output")

    text = text.strip()

    # Extract JSON from markdown code fences (```json ... ```)
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    else:
        # No fences, try to find JSON block by looking for { or [
        # Remove any text before first { or [
        json_start = min(
            (text.find("{") if text.find("{") != -1 else len(text)),
            (text.find("[") if text.find("[") != -1 else len(text)),
        )
        if json_start < len(text):
            text = text[json_start:]

    # Try parsing with demjson3 first (handles unescaped newlines better)
    try:
        import demjson3

        return demjson3.decode(text)
    except:
        pass

    # Fallback: Try standard JSON parser
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Save for debugging
        import tempfile

        debug_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        debug_file.write(f"Original error: {e}\n\n")
        debug_file.write(text)
        debug_file.close()
        raise ValueError(
            f"Failed to parse JSON. Debug saved to: {debug_file.name}"
        ) from e


def generate_l3_with_llm(l1_memory: Dict, l2_memory: Dict, llm: Any) -> Dict[str, Any]:
    """
    Use LLM to analyze L1 + L2 and generate universal patterns.

    Returns:
        {
            "universal_patterns": [...]
        }
    """
    if llm is None:
        raise ValueError("LLM is required for L3 generation")

    # Build prompt
    prompt = build_l3_prompt(l1_memory, l2_memory)

    # DEBUG: Save prompt for inspection
    import tempfile
    from pathlib import Path

    debug_dir = Path(
        "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/data/debug"
    )
    debug_dir.mkdir(exist_ok=True)
    issue_id = l1_memory.get("issue_id", "unknown")
    with open(debug_dir / f"l3_prompt_{issue_id}.txt", "w") as f:
        f.write(prompt)

    # Call LLM
    response = llm.invoke(prompt)

    # Parse response
    response_text = extract_text(response)

    # DEBUG: Save response for inspection
    with open(debug_dir / f"l3_response_{issue_id}.txt", "w") as f:
        f.write(response_text)

    # Debug: Check if response is empty
    if not response_text or not response_text.strip():
        import tempfile

        debug_file = tempfile.NamedTemporaryFile(
            mode="w", suffix="_empty_response.txt", delete=False
        )
        debug_file.write(f"Response object: {response}\n\n")
        debug_file.write(f"Response type: {type(response)}\n\n")
        debug_file.write(f"Response text: {response_text}\n")
        debug_file.close()
        raise ValueError(
            f"LLM returned empty response. Debug saved to: {debug_file.name}"
        )

    l3_data = parse_json_from_text(response_text)

    # DEBUG: Save parsed L3 data
    with open(debug_dir / f"l3_parsed_{issue_id}.json", "w") as f:
        json.dump(l3_data, f, indent=2)

    return l3_data


def build_l3_memory(
    l1_memory: Dict[str, Any],
    l2_memory: Dict[str, Any],
    llm: Any = None,
) -> Dict[str, Any]:
    """
    Build L3 universal patterns from L1 + L2.

    Falls back to empty structure if LLM generation fails.
    """
    """
    Build L3 universal patterns from L1 + L2.

    The LLM analyzes L1 problems + L2 strategies to extract:
    - Universal fix patterns that work across repos
    - Concrete examples as evidence
    - Dependency chains from enabled relationships

    Args:
        l1_memory: L1 memory with concrete problems
        l2_memory: L2 memory with repair strategies
        llm: LLM instance for analysis

    Returns:
        L3 memory with universal patterns:
        {
          "issue_id": "102",
          "repo": "flower-ai/flower",
          "workflow": ".github/workflows/framework.yml",
          "universal_patterns": [
            {
              "pattern_id": "numpy-plugin-type-removal",
              "failure_type": "type_checking",
              "failure_pattern": "...",
              "problem": "...",
              "reasoning": "...",
              "when_to_apply": "...",
              "signals": [...],
              "universal_fix": {
                "approach": "...",
                "steps": [...],
                "applies_to": [...]
              },
              "examples": [
                {"file": "...", "before": "...", "after": "..."}
              ],
              "dependent_changes": [...]
            }
          ]
        }
    """

    if llm is None:
        raise ValueError("LLM is required for L3 generation")

    # LLM analyzes L1 + L2 and generates universal patterns
    try:
        l3_data = generate_l3_with_llm(l1_memory, l2_memory, llm)
        universal_patterns = l3_data.get("universal_patterns", [])

        # DEBUG: Show L3 generation results
        if not universal_patterns:
            print("  WARNING  L3 generated 0 patterns (LLM returned empty list)")
        else:
            print(f"  OK L3 generated {len(universal_patterns)} patterns")

    except Exception as e:
        print(f"  FAIL WARNING: L3 generation failed: {e}")
        print("  Falling back to empty L3 structure")
        universal_patterns = []

    # Build final L3 structure
    l3_memory = {
        "issue_id": l1_memory.get("issue_id"),
        "repo": l1_memory.get("repo"),
        "workflow": l1_memory.get("workflow"),
        "universal_patterns": universal_patterns,
    }

    return l3_memory


# Legacy functions for backward compatibility
def normalize_l3_record(l3_record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize L3 record (kept for backward compatibility)."""
    return l3_record


def merge_l3_records(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two L3 records (kept for backward compatibility).

    For new format, merges universal_patterns by pattern_id.
    """
    if "universal_patterns" not in existing:
        return new

    if "universal_patterns" not in new:
        return existing

    # Merge patterns by pattern_id
    merged_patterns = {}

    for pattern in existing.get("universal_patterns", []):
        pid = pattern.get("pattern_id")
        if pid:
            merged_patterns[pid] = pattern

    for pattern in new.get("universal_patterns", []):
        pid = pattern.get("pattern_id")
        if pid:
            if pid in merged_patterns:
                # Merge examples and dependent_changes
                existing_pattern = merged_patterns[pid]

                # Merge examples (limit to 10)
                existing_examples = existing_pattern.get("examples", [])
                new_examples = pattern.get("examples", [])
                all_examples = existing_examples + new_examples

                # Deduplicate by file
                seen_files = set()
                unique_examples = []
                for ex in all_examples:
                    file = ex.get("file")
                    if file and file not in seen_files:
                        seen_files.add(file)
                        unique_examples.append(ex)
                        if len(unique_examples) >= 10:
                            break

                existing_pattern["examples"] = unique_examples

                # Merge dependent_changes (deduplicate)
                existing_deps = existing_pattern.get("dependent_changes", [])
                new_deps = pattern.get("dependent_changes", [])
                merged_deps = list(dict.fromkeys(existing_deps + new_deps))
                existing_pattern["dependent_changes"] = merged_deps[:10]
            else:
                merged_patterns[pid] = pattern

    return {
        "issue_id": new.get("issue_id", existing.get("issue_id")),
        "repo": new.get("repo", existing.get("repo")),
        "workflow": new.get("workflow", existing.get("workflow")),
        "universal_patterns": list(merged_patterns.values()),
    }


def create_search_document_l3(l3_record: Dict[str, Any]) -> str:
    """
    Create search document for L3 record embedding.

    Focuses on universal patterns for retrieval.
    """
    parts = []

    # Extract from all patterns
    for pattern in l3_record.get("universal_patterns", []):
        pattern_parts = []

        # Pattern ID
        if pattern.get("pattern_id"):
            pattern_parts.append(f"pattern:{pattern['pattern_id']}")

        # Failure type and pattern
        if pattern.get("failure_type"):
            pattern_parts.append(f"type:{pattern['failure_type']}")

        if pattern.get("failure_pattern"):
            pattern_parts.append(f"fails:{pattern['failure_pattern']}")

        # Problem description
        if pattern.get("problem"):
            problem_snippet = pattern["problem"][:200]
            pattern_parts.append(f"problem:{problem_snippet}")

        # Universal fix approach
        if pattern.get("universal_fix", {}).get("approach"):
            pattern_parts.append(f"fix:{pattern['universal_fix']['approach']}")

        # Signals (first 3)
        signals = pattern.get("signals", [])
        if signals:
            pattern_parts.append(f"signals:{' | '.join(signals[:3])}")

        parts.append(" | ".join(pattern_parts))

    return " || ".join(parts)


__all__ = [
    "build_l3_memory",
    "normalize_l3_record",
    "merge_l3_records",
    "create_search_document_l3",
]
