"""
Debug patch to find ALL locations where .get() is called on lists
Add this at the TOP of cibench.py to catch the error with full traceback
"""

import sys
import traceback

# Monkey-patch list to catch .get() calls
_original_list = list

class DebugList(_original_list):
    """List that logs when .get() is called (which is always wrong)"""

    def get(self, *args, **kwargs):
        print("\n" + "="*80)
        print("🐛 ERROR: .get() called on a list!")
        print("="*80)
        print("Traceback:")
        for line in traceback.format_stack():
            print(line.strip())
        print("="*80)
        print(f"List contents (first 3 items): {self[:3] if self else '[]'}")
        print("="*80 + "\n")

        # Raise the actual error
        raise AttributeError("'list' object has no attribute 'get'")

# Replace list globally
list = DebugList

print("✅ Debug patch installed - will catch .get() on list with full traceback")
