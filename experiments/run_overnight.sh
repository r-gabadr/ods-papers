#!/bin/bash
set -e
cd /home/r/z_folder/a/conda/ods-unified-v2
source /home/r/miniconda3/bin/activate new-stack
echo "Starting overnight test at $(date)"
echo "Python: $(which python)"
nohup python -u _claude/experiments/overnight_p05_data.py > _claude/experiments/overnight.log 2>&1 &
PID=$!
echo "Launched with PID=$PID"
disown $PID
sleep 12
echo "--- First lines of output ---"
head -20 _claude/experiments/overnight.log 2>/dev/null || echo "(no output yet)"
echo "--- Process check ---"
ps aux | grep overnight_p05 | grep -v grep || echo "(process not found)"
