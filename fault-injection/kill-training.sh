#!/bin/bash
set -euo pipefail

# Simulate a hard crash mid-training.
# This is the scenario: node loses power, process dies with no warning.
# Recovery must resume from the latest valid checkpoint.

echo "=== Fault Injection: Hard Kill ==="
echo "Finding training process..."

PID=$(pgrep -f "fsdp_trainer.py" | head -1)

if [ -z "$PID" ]; then
    echo "No training process found."
    exit 1
fi

STEP=$(tail -1 logs/training-*.log 2>/dev/null | grep -oP 'Step \K[0-9]+' || echo "unknown")

echo "Training PID: $PID (approx step $STEP)"
echo "Sending kill -9 (no cleanup, no checkpoint save)..."

kill -9 "$PID"

echo "Process killed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Recovery test: restart training — it should resume from the latest checkpoint."
echo "Run: make train"
