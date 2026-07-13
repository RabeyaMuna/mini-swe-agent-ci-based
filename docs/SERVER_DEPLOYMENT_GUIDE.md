# Server Deployment Guide - Mini-SWE-Agent CI-Based

Complete guide for deploying and running the mini-swe-agent CI-based system on a remote server.

---

## **Table of Contents**

1. [Server Requirements](#server-requirements)
2. [Initial Setup](#initial-setup)
3. [Configuration](#configuration)
4. [Running Evaluations](#running-evaluations)
5. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
6. [Performance Optimization](#performance-optimization)
7. [Common Issues](#common-issues)

---

## **Server Requirements**

### **Minimum Requirements:**
- **OS:** Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **CPU:** 4 cores
- **RAM:** 16GB
- **Disk:** 100GB free space
- **Network:** Outbound internet access (for API calls)

### **Recommended Requirements:**
- **CPU:** 8-16 cores
- **RAM:** 32GB+
- **Disk:** 200GB+ SSD
- **Network:** High bandwidth (1Gbps+)

### **Software Requirements:**
- Python 3.10 or 3.11
- Git 2.30+
- Docker 20.10+ (optional, for sandboxing)
- Node.js 18+ (if testing Node repos)

---

## **Initial Setup**

### **1. Connect to Server**

```bash
ssh user@your-server.com
```

### **2. Install System Dependencies**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3.10-dev \
    git curl wget build-essential \
    docker.io docker-compose \
    tmux htop
```

**CentOS/RHEL:**
```bash
sudo yum install -y \
    python310 python310-devel \
    git curl wget gcc gcc-c++ make \
    docker docker-compose \
    tmux htop
```

### **3. Configure Docker (if using)**

```bash
# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Test docker
docker run hello-world
```

### **4. Clone Repository**

```bash
cd ~
git clone https://github.com/your-username/mini-swe-agent-ci-based.git
cd mini-swe-agent-ci-based
```

### **5. Set Up Python Environment**

```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -e .

# Install additional packages
pip install datasets sentence-transformers scikit-learn
```

### **6. Verify Installation**

```bash
python3 -c "import minisweagent; print('✓ Mini-SWE-Agent installed')"
python3 -c "import sentence_transformers; print('✓ Sentence Transformers installed')"
```

---

## **Configuration**

### **1. API Keys**

Create `.env` file:

```bash
cat > .env << 'EOF'
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model Configuration
DEFAULT_LLM_MODEL=gpt-4
FALLBACK_LLM_MODEL=gpt-3.5-turbo

# Memory Configuration
MINI_SWE_AGENT_MEMORY_BACKEND=json
MEMORY_TOP_K=3

# Logging
LOG_LEVEL=INFO
EOF

chmod 600 .env  # Secure permissions
```

### **2. Memory Backend Configuration**

**Force JSON backend (recommended for servers):**

```bash
cat > .mini-swe-agent.config.yaml << 'EOF'
memory:
  backend: json
  top_k: 3
  similarity_threshold: 0.45

# Disable ChromaDB (avoids mutex locks)
use_chromadb: false

# Logging
logging:
  level: INFO
  save_logs: true
EOF
```

### **3. Build Memory Files**

```bash
# Build L1, L2, L3 memory from training data
bash scripts/run_memory_split_workflow.sh

# Verify memory files created
ls -lh data/trs/
# Should see:
#   failure_memory.json   (L1 - file-level)
#   repo_memory.json      (L2 - sequences)
#   cross_memory.json     (L3 - patterns)
```

### **4. Test Setup**

```bash
# Quick test on 3 issues
python3 scripts/run_eval.py \
    --issue-ids 113,121,123 \
    --ablation L1+L2+L3 \
    --workers 1

# Check results
ls results/L1_L2_L3/
```

---

## **Running Evaluations**

### **Method 1: Using Helper Script (Recommended)**

```bash
# Run full L1+L2+L3 evaluation
bash scripts/run_l1_l2_l3_eval.sh 4

# Arguments:
#   $1 = workers (default: 1)
#   $2 = issue IDs file (default: data/trs/eval_issue_ids.json)
```

### **Method 2: Direct Command**

```bash
# Full evaluation with 4 workers
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 4
```

### **Method 3: Using tmux (Long-Running)**

```bash
# Start tmux session
tmux new -s eval

# Run evaluation
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 8

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t eval
```

### **Method 4: Using nohup (Background)**

```bash
# Run in background
nohup python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 8 \
    > eval_l1_l2_l3.log 2>&1 &

# Check progress
tail -f eval_l1_l2_l3.log

# Check process
ps aux | grep run_eval
```

---

## **Monitoring & Troubleshooting**

### **1. Monitor Progress**

```bash
# Check running processes
ps aux | grep "run_eval\|cibench"

# Check results directory
watch -n 10 "ls -1 results/L1_L2_L3/ | wc -l"

# Monitor logs
tail -f results/L1_L2_L3/*/run_instance.log
```

### **2. Resource Monitoring**

```bash
# CPU & Memory
htop

# Disk space
df -h

# Disk I/O
iotop  # (needs sudo)

# Network
iftop  # (needs sudo)
```

### **3. Create Monitoring Script**

```bash
cat > scripts/monitor_eval.sh << 'EOF'
#!/bin/bash
# Monitor evaluation progress

while true; do
    clear
    echo "=== EVALUATION MONITOR ==="
    echo ""
    
    # Process status
    if ps aux | grep -q "[r]un_eval"; then
        echo "✓ Evaluation is RUNNING"
        ps aux | grep "[r]un_eval" | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
    else
        echo "✗ Evaluation is NOT running"
    fi
    
    echo ""
    
    # Progress
    if [ -d "results/L1_L2_L3" ]; then
        COMPLETED=$(ls -1 results/L1_L2_L3/ 2>/dev/null | wc -l)
        echo "Issues completed: $COMPLETED"
    fi
    
    echo ""
    
    # Resources
    echo "System Resources:"
    free -h | grep "Mem:" | awk '{print "  Memory:", $3, "/", $2, "used"}'
    df -h . | tail -1 | awk '{print "  Disk:", $3, "/", $2, "used (", $5, ")"}'
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    
    sleep 10
done
EOF

chmod +x scripts/monitor_eval.sh
bash scripts/monitor_eval.sh
```

---

## **Performance Optimization**

### **1. Optimize Worker Count**

```bash
# Check CPU cores
nproc

# Recommended workers = cores / 2
# Example: 16 cores → 8 workers
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 8
```

### **2. Memory Optimization**

If running out of memory:

```bash
# Reduce workers
--workers 1

# Process in batches
python3 scripts/run_eval.py \
    --issue-ids 113,121,123,102,106 \
    --ablation L1+L2+L3 \
    --workers 2
```

### **3. Disk Space Management**

```bash
# Clean old results
rm -rf results/old_runs/

# Compress completed runs
tar -czf results_backup.tar.gz results/L1_L2_L3/
```

---

## **Common Issues**

### **Issue 1: ChromaDB Mutex Lock**

**Symptom:**
```
[mutex.cc : 452] RAW: Lock blocking
```

**Solution:**
```bash
# Force JSON backend
export MINI_SWE_AGENT_MEMORY_BACKEND=json

# Or add to .env
echo "MINI_SWE_AGENT_MEMORY_BACKEND=json" >> .env

# Restart evaluation
```

### **Issue 2: Out of Memory**

**Symptom:**
```
Process killed
MemoryError
```

**Solution:**
```bash
# Reduce workers
python3 scripts/run_eval.py \
    --ablation L1+L2+L3 \
    --workers 1

# Or increase swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### **Issue 3: Docker Permission Denied**

**Symptom:**
```
permission denied while trying to connect to Docker daemon
```

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker

# Test
docker ps
```

### **Issue 4: API Rate Limits**

**Symptom:**
```
RateLimitError: Rate limit exceeded
```

**Solution:**
```bash
# Add delays between requests
# Edit in code or reduce workers

# Use fewer workers
--workers 1

# Wait and retry
```

### **Issue 5: Git Errors in Testbed**

**Symptom:**
```
fatal: not a git repository
```

**Solution:**
```bash
# Clean testbed directories
rm -rf /tmp/swe_agent_*

# Restart evaluation
```

---

## **Running Ablation Studies**

### **Complete Ablation Study Workflow**

```bash
# 1. BASELINE (no memory)
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation BASELINE \
    --workers 8

# 2. L1 (file-level memory)
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1 \
    --workers 8

# 3. L1+L2 (file + sequences)
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2 \
    --workers 8

# 4. L1+L2+L3 (full memory)
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 8
```

### **Automated Ablation Script**

```bash
cat > scripts/run_all_ablations.sh << 'EOF'
#!/bin/bash
set -e

WORKERS=${1:-8}
ISSUES_FILE=${2:-data/trs/eval_issue_ids.json}

for ablation in BASELINE L1 L1+L2 L1+L2+L3; do
    echo "========================================"
    echo "Running ablation: $ablation"
    echo "========================================"
    
    python3 scripts/run_eval.py \
        --issue-ids-file "$ISSUES_FILE" \
        --ablation "$ablation" \
        --workers "$WORKERS"
    
    echo "✓ $ablation complete"
    echo ""
done

echo "All ablations complete!"
EOF

chmod +x scripts/run_all_ablations.sh

# Run all ablations
bash scripts/run_all_ablations.sh 8
```

---

## **Results Collection**

### **After Evaluation Completes:**

```bash
# Check results structure
ls -lh results/

# Should see:
#   results/BASELINE/
#   results/L1/
#   results/L1_L2/
#   results/L1_L2_L3/

# Collect predictions
cat results/L1_L2_L3/preds.json

# Evaluate metrics (if evaluation script exists)
python3 scripts/evaluate_preds.py results/L1_L2_L3/preds.json

# Compare ablations
python3 scripts/compare_ablations.py \
    --baseline results/BASELINE/preds.json \
    --l1 results/L1/preds.json \
    --l1-l2 results/L1_L2/preds.json \
    --l1-l2-l3 results/L1_L2_L3/preds.json
```

---

## **Backup & Download Results**

```bash
# Compress results
tar -czf results_$(date +%Y%m%d).tar.gz results/

# Download from server (run on local machine)
scp user@server:/path/to/results_*.tar.gz ./

# Or use rsync
rsync -avz user@server:/path/to/results/ ./results_backup/
```

---

## **Server-Specific Tips**

### **1. Use tmux for Long-Running Jobs**

```bash
# Create session
tmux new -s eval_session

# Run evaluation
bash scripts/run_l1_l2_l3_eval.sh 8

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t eval_session
```

### **2. Schedule with cron**

```bash
# Edit crontab
crontab -e

# Add job (runs daily at 2 AM)
0 2 * * * cd /home/user/mini-swe-agent-ci-based && /home/user/mini-swe-agent-ci-based/.venv/bin/python3 scripts/run_eval.py --issue-ids-file data/trs/eval_issue_ids.json --ablation L1+L2+L3 --workers 8 >> /home/user/eval_cron.log 2>&1
```

### **3. Auto-Restart on Failure**

```bash
cat > scripts/run_with_retry.sh << 'EOF'
#!/bin/bash
MAX_RETRIES=3
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY+1))/$MAX_RETRIES"
    
    python3 scripts/run_eval.py \
        --issue-ids-file data/trs/eval_issue_ids.json \
        --ablation L1+L2+L3 \
        --workers 8
    
    if [ $? -eq 0 ]; then
        echo "✓ Success!"
        exit 0
    else
        echo "✗ Failed, retrying..."
        RETRY=$((RETRY+1))
        sleep 60
    fi
done

echo "✗ Failed after $MAX_RETRIES attempts"
exit 1
EOF

chmod +x scripts/run_with_retry.sh
```

---

## **Summary**

### **Quick Start:**
```bash
# 1. Setup
ssh server
git clone <repo>
cd mini-swe-agent-ci-based
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env with API keys

# 3. Build memory
bash scripts/run_memory_split_workflow.sh

# 4. Run evaluation
bash scripts/run_l1_l2_l3_eval.sh 8

# 5. Monitor
bash scripts/monitor_eval.sh
```

### **Key Points:**
- ✅ Use JSON backend (not ChromaDB) on servers
- ✅ Use tmux for long-running jobs
- ✅ Monitor resources (htop, df -h)
- ✅ Workers = CPU cores / 2
- ✅ Backup results regularly

---

**Need help?** Check logs in `results/L1_L2_L3/*/run_instance.log`
