#!/bin/bash
set -euo pipefail

# Simulate Slurm preemption — sends SIGTERM then waits.
# The trainer should catch SIGTERM, save a checkpoint, and exit gracefully.
# This is what happens when Slurm preempts your job for a higher-priority one.

echo "=== Fault Injection: Slurm Preemption (SIGTERM) ==="
echo "Finding training process..."

PID=$(pgrep -f "fsdp_trainer.py" | head -1)

if [ -z "$PID" ]; then
    echo "No training process found."
    exit 1
fi

echo "Training PID: $PID"
echo "Sending SIGTERM (graceful shutdown)..."

kill -TERM "$PID"

echo "Waiting for process to save checkpoint and exit..."
TIMEOUT=120
ELAPSED=0
while kill -0 "$PID" 2>/dev/null; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Process did not exit within ${TIMEOUT}s — sending kill -9"
        kill -9 "$PID"
        break
    fi
done

echo "Process exited after ${ELAPSED}s"
echo ""
echo "Verify: check that a checkpoint was saved before exit."
echo "Then restart: make train (should resume from that checkpoint)"
