# Debug Memory Error - Find the Exact Location

The error "'list' object has no attribute 'get'" keeps appearing even after multiple fixes.

## **On Server - Get the Full Traceback**

```bash
cd ~/Documents/rabeya/mini-swe-agent-ci-based

# Run with full traceback
python3 scripts/run_eval.py \
    --issue-ids 71 \
    --ablation L1+L2+L3 \
    --workers 1 2>&1 | grep -A50 "'list' object"

# Or check the detailed log
tail -100 results/L1_L2_L3/cibench.log | grep -B20 -A20 "list' object"
```

## **Alternative: Add Debug Prints**

Edit `src/minisweagent/run/benchmarks/utils/ci_memory_system.py`:

At line 493 (in the loop), add:

```python
for mem in memories:
    # DEBUG: Print type
    if not isinstance(mem, dict):
        print(f"❌ ERROR: mem is {type(mem)}, not dict!")
        print(f"   Value: {mem}")
        import traceback
        traceback.print_stack()
    
    stage_info = _map_memory_to_validation_stage(mem, validation_sequence)
```

Then re-run and the exact location will print!

## **Quick Fix: Wrap the Entire Loop**

Edit line 493 in `ci_memory_system.py`:

```python
for mem in memories:
    # CRITICAL FIX: Validate mem is a dict
    if not isinstance(mem, dict):
        logger.warning(f"[organize_by_stage] Skipping non-dict memory item: {type(mem)}")
        continue
    
    stage_info = _map_memory_to_validation_stage(mem, validation_sequence)
```

## **Nuclear Option: Check ALL .get() Calls**

Add this at the TOP of `ci_memory_system.py` after imports:

```python
# Monkey-patch for debugging
_original_list = list

class DebugList(_original_list):
    def get(self, *args, **kwargs):
        import traceback
        print("❌ ERROR: Calling .get() on a list!")
        print("Traceback:")
        traceback.print_stack()
        raise AttributeError("'list' object has no attribute 'get'")

# Uncomment to enable debugging:
# list = DebugList
```

This will show EXACTLY where the bug is!

## **Most Likely Culprit**

Based on the flow, the error is probably in line 493-500 range:

```python
# Line 493
for mem in memories:  # ← If 'memories' contains a list instead of dict
    stage_info = _map_memory_to_validation_stage(mem, validation_sequence)
    stage_name = stage_info["validates"]
    stage_order = stage_info["order"]
    
    if len(by_stage) < 3:
        logger.info(f"Memory: validation_cmd='{mem.get('validation_cmd', '')[:50]}'")
        # ↑ THIS LINE 500 - if mem is a list, ERROR!
```

## **The Fix**

```bash
# On server, edit the file:
nano +493 src/minisweagent/run/benchmarks/utils/ci_memory_system.py

# Change line 493-500 to:
for mem in memories:
    # Validate mem is a dict
    if not isinstance(mem, dict):
        logger.warning(f"[organize_by_stage] Skipping invalid memory item (type={type(mem).__name__})")
        continue
    
    stage_info = _map_memory_to_validation_stage(mem, validation_sequence)
    stage_name = stage_info["validates"]
    stage_order = stage_info["order"]
    
    # Debug: log first few mappings
    if len(by_stage) < 3:
        logger.info(f"Memory: validation_cmd='{mem.get('validation_cmd', '')[:50]}' -> stage='{stage_name}' order={stage_order}")
```

Save and re-run!
