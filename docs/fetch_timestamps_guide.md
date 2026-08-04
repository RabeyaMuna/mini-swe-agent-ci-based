# Quick Guide: Fetching Commit Timestamps

## Overview

The [`fetch_commit_timestamps.py`](../scripts/fetch_commit_timestamps.py) script fetches commit dates from GitHub for all issues in your CI benchmark.

## Basic Usage

### 1. Simple Run (No Authentication)

```bash
python scripts/fetch_commit_timestamps.py
```

**Rate Limit:** 60 requests/hour (enough for small datasets)

**Time Required:** 
- 126 issues × 0.5s delay = ~1 minute
- Plus API response time = ~5 minutes total

### 2. With GitHub Token (Recommended for Large Datasets)

```bash
# Option A: Pass token as argument
python scripts/fetch_commit_timestamps.py \
    --github-token ghp_yourTokenHere

# Option B: Use environment variable (more secure)
export GITHUB_TOKEN=ghp_yourTokenHere
python scripts/fetch_commit_timestamps.py
```

**Rate Limit:** 5,000 requests/hour (plenty for any dataset)

**How to get a token:**
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "CI Benchmark Timestamp Fetcher"
4. Select scopes: **Only need `public_repo` or no scopes at all!**
5. Copy the token (starts with `ghp_`)

### 3. Filter by Repositories

```bash
# Only fetch for specific repos
python scripts/fetch_commit_timestamps.py --repos agno,flower
```

### 4. Resume Interrupted Fetch

```bash
# If fetch was interrupted, resume from cache
python scripts/fetch_commit_timestamps.py --resume
```

The script automatically saves progress every 10 successful fetches, so you won't lose much if interrupted.

---

## Output

### Default Output Location

```
data/commit_timestamps.json
```

### Output Format

```json
{
  "1": {
    "timestamp": "2023-03-15T14:32:00Z",
    "repo": "huggingface/diffusers",
    "sha": "2c06ffa4c9d2c37846c60ad75899b4d72f214ff9",
    "repo_owner": "huggingface",
    "repo_name": "diffusers",
    "author_name": "John Doe",
    "author_email": "john@example.com",
    "commit_message": "Fix linting issues in examples"
  },
  "2": {
    "timestamp": "2021-05-20T10:15:30Z",
    "repo": "huggingface/diffusers",
    ...
  }
}
```

### Example Output

```
Fetching Commit Timestamps from GitHub

GitHub API Rate Limit:
  Limit: 5000 requests/hour
  Remaining: 4998
  Resets at: 15:32:00

Loading dataset: ci-benchmark-user/ci-repair-bench
Loaded 567 issues

Fetching timestamps for 567 issues...
Delay: 0.5s between requests

⠋ Fetching timestamps... ━━━━━━━━━━━━━━━━━━━━━━━━ 100% 567/567

Fetch Complete!

┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Status            ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Issues      │   567 │
│ Successfully      │   562 │
│ Fetched           │       │
│ Failed            │     5 │
│ Skipped (cached)  │     0 │
│ Total in Cache    │   567 │
└───────────────────┴───────┘

Timestamp Range:
  Earliest: 2015-01-10
  Latest: 2024-03-22

Saved to: data/commit_timestamps.json
```

---

## Advanced Options

### Custom Output Path

```bash
python scripts/fetch_commit_timestamps.py \
    --output data/my_custom_timestamps.json
```

### Adjust Request Delay

```bash
# Faster (but respect rate limits!)
python scripts/fetch_commit_timestamps.py --delay 0.2

# Slower (more conservative)
python scripts/fetch_commit_timestamps.py --delay 1.0
```

### Specific Dataset

```bash
python scripts/fetch_commit_timestamps.py \
    --dataset-name "your-org/your-dataset"
```

---

## Troubleshooting

### Rate Limit Exceeded

**Error:** `GitHub API rate limit exceeded`

**Solution:**
1. Use a GitHub token: `--github-token YOUR_TOKEN`
2. Or wait for rate limit reset (script tells you when)

### 404 Not Found

**Cause:** Repository or commit was deleted

**Impact:** That issue will be marked with `"timestamp": null`

**Action:** Continue - the script handles this gracefully

### Timeout Errors

**Cause:** Network issues or slow GitHub API

**Solution:** 
- Script auto-retries 3 times
- Use `--resume` to continue from where it left off

### Invalid SHA

**Cause:** Malformed commit SHA in dataset

**Impact:** Issue marked with `"timestamp": null`

**Action:** Review failed issues after completion

---

## Integration with Other Scripts

### 1. Use with Temporal Split

```bash
# Step 1: Fetch timestamps
python scripts/fetch_commit_timestamps.py

# Step 2: Use in temporal split
python scripts/prepare_memory_train_test_split.py \
    --repos agno,flower,camel \
    --temporal-split \
    --timestamps data/commit_timestamps.json \
    --output-dir data/trs_temporal
```

### 2. Use with Leakage Analysis

```bash
# Step 1: Fetch timestamps
python scripts/fetch_commit_timestamps.py

# Step 2: Analyze leakage
python scripts/analyze_temporal_leakage.py \
    --timestamp-cache data/commit_timestamps.json \
    --memory-ids data/trs/memory_issue_ids.json \
    --eval-ids data/trs/eval_issue_ids.json
```

---

## Performance & Timing

### Without GitHub Token (60/hour limit)

| Issues | Time Estimate |
|--------|---------------|
| 60     | ~1 minute     |
| 120    | ~3 minutes (need to wait for rate reset) |
| 567    | ~5 hours (with waits) |

**Recommendation:** Use token for >60 issues

### With GitHub Token (5000/hour limit)

| Issues | Time Estimate |
|--------|---------------|
| 60     | ~1 minute     |
| 120    | ~1.5 minutes  |
| 567    | ~5 minutes    |
| 5000   | ~45 minutes   |

**Recommendation:** Standard approach for any size dataset

---

## Script Features

OK **Auto-caching** - Saves progress every 10 fetches  
OK **Resume support** - Continue from interruption  
OK **Rate limit handling** - Auto-waits when limit hit  
OK **Retry logic** - 3 retries with exponential backoff  
OK **Error handling** - Graceful failure for missing commits  
OK **Progress tracking** - Real-time progress bar  
OK **Rich metadata** - Stores author, message, etc.  
OK **Repo filtering** - Fetch only what you need  

---

## What Gets Stored

For each issue, the script fetches and stores:

- OK **Commit timestamp** (ISO 8601 format)
- OK **Repository** (owner/name)
- OK **Commit SHA**
- OK **Author name**
- OK **Author email**
- OK **Commit message** (first line, truncated to 100 chars)

This extra metadata can be useful for:
- Analyzing commit patterns
- Grouping by author
- Understanding fix context

---

## Example Workflow

```bash
# 1. Create GitHub token (optional but recommended)
# Go to: https://github.com/settings/tokens
export GITHUB_TOKEN=ghp_yourTokenHere

# 2. Fetch timestamps for your repos
python scripts/fetch_commit_timestamps.py \
    --repos agno,flower,camel \
    --output data/commit_timestamps.json

# 3. Check what was fetched
cat data/commit_timestamps.json | jq 'length'  # Total count
cat data/commit_timestamps.json | jq '.[0]'     # First entry

# 4. Check date range
cat data/commit_timestamps.json | \
  jq -r '.[] | .timestamp' | \
  sort | \
  head -1  # Earliest

cat data/commit_timestamps.json | \
  jq -r '.[] | .timestamp' | \
  sort | \
  tail -1  # Latest

# 5. Use in temporal split
python scripts/prepare_memory_train_test_split.py \
    --temporal-split \
    --timestamps data/commit_timestamps.json \
    --output-dir data/trs_temporal
```

---

## FAQ

### Q: Do I need to re-fetch if I add more issues?

**A:** No! Use `--resume` flag and it will only fetch new issues:

```bash
python scripts/fetch_commit_timestamps.py --resume
```

### Q: What if GitHub is down?

**A:** Script has 3 retries with backoff. If still failing, use `--resume` later.

### Q: Can I use this for private repos?

**A:** Yes, but you MUST use a GitHub token with appropriate permissions.

### Q: Will this work for very old commits (10+ years)?

**A:** Yes! Git timestamps are permanent and immutable.

### Q: What if I hit rate limit?

**A:** Script auto-waits for reset. Or use `--github-token` for 5000/hour limit.

---

## Summary

**Simplest command:**
```bash
python scripts/fetch_commit_timestamps.py
```

**Recommended command:**
```bash
export GITHUB_TOKEN=ghp_yourTokenHere
python scripts/fetch_commit_timestamps.py --repos agno,flower,camel
```

**Output:** `data/commit_timestamps.json` with all commit dates

**Next:** Use with `--temporal-split` in your memory/eval split script!
