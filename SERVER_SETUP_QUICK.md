# Quick Server Setup & Fix Guide

## **CRITICAL ERRORS FOUND ON YOUR SERVER**

### **Error 1: No Embedding Model (STOPS L1+L2+L3!)**
```
[Memory] WARNING: No embedding model available
memory retrieval disabled
```

### **Error 2: List Object Bug**
```
'list' object has no attribute 'get'
```

---

## **IMMEDIATE FIX (Run on Server)**

```bash
# SSH to your server
ssh ubuntu@your-server

# Navigate to project
cd ~/Documents/rabeya/mini-swe-agent-ci-based

# Activate virtual environment (if using one)
source .venv/bin/activate  # or skip if not using venv

# Install ALL dependencies (recommended)
pip install -e .

# OR install just the missing critical ones
pip install sentence-transformers accelerate

# Verify installation
python3 scripts/verify_installation.py
```

---

## **After Installing Dependencies**

### **Kill Current Process:**
```bash
# Press Ctrl+C on the running evaluation

# Or kill from another terminal:
pkill -9 -f "cibench"
```

### **Re-run Evaluation:**
```bash
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 4
```

---

## **What Was Wrong**

### **1. Missing sentence-transformers**
- **Impact:** Memory retrieval completely disabled!
- **What L1+L2+L3 does:** Computes embeddings to find similar problems
- **Without it:** Returns empty results (no memory used at all!)
- **Fix:** `pip install sentence-transformers accelerate`

### **2. accelerate package missing**
- **Impact:** sentence-transformers can't load models
- **Fix:** `pip install accelerate`

---

## **Expected Output After Fix**

You should see:
```
✓ [Memory] Loaded memory banks: L1=173, L2=37, L3=119
✓ [Memory] Using sentence-transformers model: all-MiniLM-L6-v2
✓ [L1 Retrieval] Top similarity: 0.8542
✓ [L1 Pipeline] similar=10 → expanded=13 → clustered=11
```

Instead of:
```
✗ [Memory] WARNING: No embedding model available
✗ Cosine similarity will return 0.0 — memory retrieval disabled
```

---

## **Full Server Setup (If Starting Fresh)**

```bash
# 1. System packages
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
    git build-essential

# 2. Clone repo
git clone <your-repo-url>
cd mini-swe-agent-ci-based

# 3. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 4. Install project
pip install --upgrade pip
pip install -e .

# 5. Install CRITICAL dependencies
pip install sentence-transformers accelerate scikit-learn

# 6. Configure
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
MINI_SWE_AGENT_MEMORY_BACKEND=json
EOF

# 7. Build memory
bash scripts/run_memory_split_workflow.sh

# 8. Test
python3 scripts/run_eval.py \
    --issue-ids 113,121,123 \
    --ablation L1+L2+L3 \
    --workers 1

# 9. Run full evaluation
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 4
```

---

## **Performance Tips for Server**

### **Check Server Resources:**
```bash
# CPU cores
nproc

# Available RAM
free -h

# Disk space
df -h
```

### **Optimize Workers:**
```bash
# Workers = CPU cores / 2
# Example: 16 cores → 8 workers

python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 8
```

### **Run in Background:**
```bash
# Option 1: tmux (recommended)
tmux new -s eval
python3 scripts/run_eval.py --issue-ids-file data/trs/eval_issue_ids.json --ablation L1+L2+L3 --workers 8
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t eval

# Option 2: nohup
nohup python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 8 \
    > eval.log 2>&1 &

# Monitor
tail -f eval.log
```

---

## **Troubleshooting**

### **Check if embeddings work:**
```bash
python3 << 'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("test")
print(f"✓ Embedding size: {len(embedding)}")
EOF
```

Expected output: `✓ Embedding size: 384`

### **Check memory files:**
```bash
ls -lh data/trs/*.json
```

Should see:
- `failure_memory.json` (L1)
- `repo_memory.json` (L2)  
- `cross_memory.json` (L3)

### **Monitor progress:**
```bash
# Count completed issues
watch -n 10 "ls -1 results/L1_L2_L3/ | wc -l"

# Check logs
tail -f results/L1_L2_L3/cibench.log
```

---

## **Next Steps After Fix**

1. **Install sentence-transformers**: `pip install sentence-transformers accelerate`
2. **Kill current run**: `Ctrl+C` or `pkill -9 -f cibench`
3. **Re-run**: `python3 scripts/run_eval.py --issue-ids-file data/trs/eval_issue_ids.json --ablation L1+L2+L3 --workers 4`
4. **Monitor**: Check logs for "Loaded memory banks" and "L1 Pipeline" messages
5. **Wait**: 89 issues × 3 min/issue ÷ 4 workers = ~67 minutes

---

## **Verification Commands**

```bash
# 1. Check sentence-transformers installed
pip list | grep sentence

# 2. Check it loads
python3 -c "from sentence_transformers import SentenceTransformer; print('OK')"

# 3. Check memory files exist
ls data/trs/*.json

# 4. Check evaluation is running
ps aux | grep cibench

# 5. Check results being created
ls -lh results/L1_L2_L3/
```

---

## **Contact/Issues**

If you still see:
- `No embedding model available` → sentence-transformers not installed correctly
- `'list' object has no attribute 'get'` → Bug in code, needs investigation
- `ChromaDB mutex lock` → Switch to JSON backend (already configured)

**The #1 critical issue is installing sentence-transformers!**
