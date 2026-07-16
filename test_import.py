#!/usr/bin/env python3
"""
Quick test to verify the fix works after cache clear.
"""
import sys
sys.path.insert(0, '/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/src')

print("🧪 Testing import and function signature...")

try:
    from minisweagent.run.benchmarks.utils.problem_validator import filter_and_validate_problems
    print("[OK] problem_validator imported successfully")

    # Check function signature
    import inspect
    sig = inspect.signature(filter_and_validate_problems)
    params = list(sig.parameters.keys())

    print(f"[OK] Function parameters: {params}")

    if 'repo_path' in params:
        print("[OK] Function accepts 'repo_path' parameter")
    else:
        print("[FAIL] Function missing 'repo_path' parameter")
        sys.exit(1)

    # Try to import and check cibench
    print("\n🧪 Checking cibench.py...")
    with open('src/minisweagent/run/benchmarks/cibench.py') as f:
        content = f.read()

    if 'repo_path=str(testbed_path)' in content:
        print("[OK] cibench.py uses testbed_path correctly")
    else:
        print("[FAIL] cibench.py not using testbed_path")
        sys.exit(1)

    if 'agent.environment.repo_path' in content:
        # Check if it's only in comments or imports
        lines_with_agent_env = [line for line in content.split('\n') if 'agent.environment' in line and not line.strip().startswith('#') and 'import' not in line.lower()]
        if lines_with_agent_env:
            print(f"[FAIL] Found agent.environment in active code: {lines_with_agent_env}")
            sys.exit(1)
        else:
            print("[OK] No active agent.environment references")
    else:
        print("[OK] No agent.environment references")

    print("\n All checks passed! The fix is ready.")
    print("\nNext steps:")
    print("   1. Make sure no cibench processes are running")
    print("   2. Re-run your cibench command")
    print("   3. The AttributeError should be gone!")

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
