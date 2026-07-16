"""
Example: Detecting and Merging Duplicate File Patches

This example shows how the patch merger handles multiple patches for the same file,
which is common when the memory-based CI repair system identifies multiple problems
affecting the same files.
"""

from minisweagent.run.benchmarks.utils.patch_merger import (
    detect_duplicate_patches,
    merge_duplicate_patches,
    validate_patch,
)

# Example: Agent generated multiple patches for the same file
# (This happens when L2 memory returns multiple problems affecting firecrawl.py)
diff_with_duplicates = """diff --git a/libs/agno/agno/tools/firecrawl.py b/libs/agno/agno/tools/firecrawl.py
index aec7b2f03..238eea64e 100644
--- a/libs/agno/agno/tools/firecrawl.py
+++ b/libs/agno/agno/tools/firecrawl.py
@@ -6,7 +6,7 @@ from agno.tools import Toolkit
 from agno.utils.log import logger

 try:
-    from firecrawl import FirecrawlApp, ScrapeOptions  # type: ignore[attr-defined]
+    from firecrawl import FirecrawlApp, V1ScrapeOptions  # type: ignore[attr-defined]
 except ImportError:
     raise ImportError("`firecrawl-py` not installed. Please install using `pip install firecrawl-py`")

diff --git a/libs/agno/agno/tools/firecrawl.py b/libs/agno/agno/tools/firecrawl.py
index 238eea64e..5f3d9a1e9 100644
--- a/libs/agno/agno/tools/firecrawl.py
+++ b/libs/agno/agno/tools/firecrawl.py
@@ -104,7 +104,7 @@ class FirecrawlTools(Toolkit):
         if self.limit or limit:
             params["limit"] = self.limit or limit
         if self.formats:
-            params["scrape_options"] = ScrapeOptions(formats=self.formats)  # type: ignore
+            params["scrape_options"] = V1ScrapeOptions(formats=self.formats)  # type: ignore

         params["poll_interval"] = self.poll_interval

diff --git a/libs/agno/agno/tools/firecrawl.py b/libs/agno/agno/tools/firecrawl.py
index 5f3d9a1e9..892c4f2a1 100644
--- a/libs/agno/agno/tools/firecrawl.py
+++ b/libs/agno/agno/tools/firecrawl.py
@@ -132,7 +132,7 @@ class FirecrawlTools(Toolkit):
         if self.limit or limit:
             params["limit"] = self.limit or limit
         if self.formats:
-            params["scrape_options"] = ScrapeOptions(formats=self.formats)  # type: ignore
+            params["scrape_options"] = V1ScrapeOptions(formats=self.formats)  # type: ignore
         if self.search_params:
             params.update(self.search_params)
"""

print("=" * 80)
print("EXAMPLE: Multiple Patches for Same File (Causes Merge Conflicts)")
print("=" * 80)

# Step 1: Detect duplicates
print("\n1. Detecting duplicate patches...")
duplicates = detect_duplicate_patches(diff_with_duplicates)

for file_path, count in duplicates.items():
    status = "[WARN] DUPLICATE" if count > 1 else "[OK]"
    print(f"   {status} - {file_path}: {count} patch(es)")

# Step 2: Validate (should fail with duplicates)
print("\n2. Validating original patch...")
is_valid, message = validate_patch(diff_with_duplicates)
print(f"   Valid: {is_valid}")
print(f"   Message: {message}")

# Step 3: Merge duplicates
print("\n3. Merging duplicate patches...")
print("   (Applying patches sequentially in temporary git repo)")
try:
    merged_diff = merge_duplicate_patches(diff_with_duplicates)
    print("   [OK] Successfully merged!")

    # Show result
    print("\n4. Result:")
    print("   Original: 3 separate patches for same file")
    print("   Merged:   1 unified patch with all changes")

    # Validate merged patch
    is_valid_merged, message_merged = validate_patch(merged_diff)
    print(f"\n5. Validation of merged patch:")
    print(f"   Valid: {is_valid_merged}")
    print(f"   Message: {message_merged}")

    # Show snippet of merged diff
    print("\n6. Merged diff (first 500 chars):")
    print("   " + merged_diff[:500].replace("\n", "\n   "))

except Exception as e:
    print(f"   [FAIL] Merge failed: {e}")

print("\n" + "=" * 80)
print("This merge happens automatically in cibench.py when saving predictions!")
print("=" * 80)
