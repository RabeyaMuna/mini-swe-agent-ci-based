#!/usr/bin/env python3
"""
Verify Mini-SWE-Agent Installation
Checks all critical dependencies are installed correctly.
"""

import sys
from typing import List, Tuple

def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name}"
    except ImportError as e:
        return False, f"✗ {package_name or module_name}: {e}"

def check_sentence_transformers() -> Tuple[bool, str]:
    """Check if sentence-transformers works correctly."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode("test")
        return True, f"✓ sentence-transformers (embedding size: {len(embedding)})"
    except Exception as e:
        return False, f"✗ sentence-transformers: {e}"

def main():
    print("=" * 60)
    print("Mini-SWE-Agent Installation Verification")
    print("=" * 60)
    print()

    checks: List[Tuple[bool, str]] = []

    # Core dependencies
    print("Checking core dependencies...")
    checks.append(check_import("yaml", "pyyaml"))
    checks.append(check_import("requests"))
    checks.append(check_import("jinja2"))
    checks.append(check_import("pydantic"))
    checks.append(check_import("litellm"))
    checks.append(check_import("rich"))
    checks.append(check_import("dotenv", "python-dotenv"))
    checks.append(check_import("typer"))
    checks.append(check_import("datasets"))
    checks.append(check_import("openai"))

    # Memory & Embeddings (CRITICAL)
    print("\nChecking memory & embedding dependencies (CRITICAL for L1+L2+L3)...")
    checks.append(check_import("sklearn", "scikit-learn"))
    checks.append(check_import("numpy"))
    checks.append(check_import("accelerate"))
    checks.append(check_sentence_transformers())

    # Optional
    print("\nChecking optional dependencies...")
    checks.append(check_import("chromadb"))
    checks.append(check_import("fastembed"))

    # Mini-SWE-Agent itself
    print("\nChecking mini-swe-agent...")
    checks.append(check_import("minisweagent"))

    # Print results
    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)

    passed = 0
    failed = 0
    critical_failed = []

    for success, message in checks:
        print(message)
        if success:
            passed += 1
        else:
            failed += 1
            # Mark critical failures
            if any(x in message for x in ["sentence-transformers", "accelerate", "scikit-learn", "numpy"]):
                critical_failed.append(message)

    print()
    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if critical_failed:
        print()
        print("🚨 CRITICAL FAILURES:")
        for msg in critical_failed:
            print(f"  {msg}")
        print()
        print("Fix with:")
        print("  pip install sentence-transformers accelerate scikit-learn numpy")
        print()
        return 1

    if failed > 0:
        print()
        print("⚠️  Some optional dependencies missing (non-critical)")
        print("   System will work but some features may be unavailable")
        print()
        return 0

    print()
    print("✅ All dependencies installed correctly!")
    print()
    print("Ready to run:")
    print("  python3 scripts/run_eval.py --issue-ids-file data/trs/eval_issue_ids.json --ablation L1+L2+L3 --workers 4")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
