#!/usr/bin/env python3
"""
Test all fixes:
1. CI log JSON parsing with malformed JSON
2. CIMemorySystem.plugin property access
3. Problem statement includes repair plan and memory guidance
"""

import sys
sys.path.insert(0, 'src')

def test_json_cleaning():
    """Test JSON cleaning function"""
    from minisweagent.run.benchmarks.utils.ci_log_analyzer import _clean_malformed_json

    # Test 1: Markdown fences
    malformed = '```json\n{"key": "value"}\n```'
    cleaned = _clean_malformed_json(malformed)
    assert '```' not in cleaned
    print("✅ Test 1: Markdown fence removal")

    # Test 2: Trailing commas
    malformed = '{"key": "value",}'
    cleaned = _clean_malformed_json(malformed)
    assert cleaned == '{"key": "value"}'
    print("✅ Test 2: Trailing comma removal")

    # Test 3: Extra text before/after JSON
    malformed = 'Here is the JSON: {"key": "value"} done'
    cleaned = _clean_malformed_json(malformed)
    assert cleaned == '{"key": "value"}'
    print("✅ Test 3: Extra text removal")

    print("\n✅ All JSON cleaning tests passed!")


def test_plugin_property():
    """Test CIMemorySystem.plugin property"""
    from minisweagent.run.benchmarks.utils.ci_memory_system import CIMemorySystem

    memory_system = CIMemorySystem.create(
        memory_root='data/trs',
        memory_enabled=True,
        memory_top_k=3,
        memory_ablation_levels='L1+L2+L3'
    )

    # Test property exists
    assert hasattr(memory_system, 'plugin'), "Missing 'plugin' property"
    print("✅ CIMemorySystem.plugin property exists")

    # Test property is accessible
    plugin = memory_system.plugin
    assert plugin is not None, "plugin is None"
    print("✅ CIMemorySystem.plugin is accessible")

    # Test generate_repair_plan method exists
    assert hasattr(plugin, 'generate_repair_plan'), "Missing generate_repair_plan method"
    print("✅ MemoryPlugin.generate_repair_plan method exists")

    print("\n✅ All plugin property tests passed!")


def test_problem_statement_sections():
    """Test that problem statement includes all necessary sections"""
    from minisweagent.run.benchmarks.utils.ci_context import build_problem_statement

    # Mock context with all sections
    context = {
        "repo": "test/repo",
        "sha_fail": "abc123",
        "workflow_name": "CI",
        "overall_error_types": ["Type Error"],
        "overall_failure_reasons": ["mypy failed"],
        "failed_jobs": [{"job": "test", "step": "lint", "command": "mypy"}],
        "effected_files": [{"file": "test.py", "reason": "type error"}],
        "workflow_profile": {
            "validation_sequence": [
                {"order": 1, "validates": "Type checking", "validation_cmd": "mypy"}
            ]
        }
    }

    # Mock memory with guidance
    memory = {
        "enabled": True,
        "llm_selection": {
            "use_memory": True,
            "guidance_document": {
                "confidence": "high",
                "confidence_reason": "Strong pattern match",
                "diagnosis": "Type annotation error",
                "primary_files": [
                    {"file": "test.py", "reason": "Invalid type", "fix": "Use proper type"}
                ],
                "additional_files": [
                    {"file": "other.py", "reason": "Hidden failure", "fix": "Fix after test.py"}
                ],
                "fix_approach": ["Step 1: Fix type", "Step 2: Verify"]
            }
        },
        "repair_plan": {
            "problem_statement": "Multi-problem CI failure",
            "root_causes": [
                {"cause": "Type error", "evidence": "mypy log", "validation_stage": "lint"}
            ],
            "repair_steps": [
                {
                    "step_id": 1,
                    "what_to_fix": "Type annotation",
                    "where_to_fix": "test.py:42",
                    "how_to_fix": "Replace DTypeLike with np.dtype",
                    "why_this_fixes": "Proper public API",
                    "verify_by": "mypy test.py"
                }
            ],
            "verification_order": [1],
            "estimated_complexity": "medium"
        }
    }

    statement = build_problem_statement(context, memory)

    # Check essential sections
    assert "# CI Failure Report" in statement, "Missing header"
    print("✅ Has header")

    assert "## Why the CI Failed" in statement, "Missing failure reasons"
    print("✅ Has failure reasons")

    assert "## Memory Context" in statement, "Missing memory context"
    print("✅ Has memory context")

    assert "### Primary files to inspect first" in statement, "Missing primary files"
    print("✅ Has primary files")

    assert "### Files to fix — including those NOT in the log" in statement, "Missing additional files"
    print("✅ Has additional/hidden files")

    assert "## Suggested Repair Plan" in statement, "Missing repair plan"
    print("✅ Has repair plan")

    assert "### Root Causes" in statement, "Missing root causes"
    print("✅ Has root causes")

    assert "### Repair Steps (in order)" in statement, "Missing repair steps"
    print("✅ Has repair steps")

    # Check critical content
    assert "other.py" in statement, "Missing hidden file from additional_files"
    print("✅ Hidden files included in statement")

    assert "Type annotation" in statement, "Missing repair step detail"
    print("✅ Repair step details included")

    print("\n✅ All problem statement tests passed!")
    print(f"\n📄 Problem statement length: {len(statement)} chars")

    # Show excerpt
    print("\n--- EXCERPT ---")
    lines = statement.split('\n')
    for i, line in enumerate(lines):
        if 'Memory Context' in line or 'Repair Plan' in line:
            print('\n'.join(lines[i:min(i+10, len(lines))]))
            break


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ALL FIXES")
    print("=" * 80)

    try:
        print("\n1. Testing JSON Cleaning...")
        test_json_cleaning()

        print("\n2. Testing Plugin Property...")
        test_plugin_property()

        print("\n3. Testing Problem Statement...")
        test_problem_statement_sections()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED! System is ready.")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
