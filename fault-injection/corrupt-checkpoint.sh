#!/bin/bash
set -euo pipefail

# Simulate checkpoint corruption.
# The checkpoint writer should detect this and fall back to the previous valid checkpoint.

CHECKPOINT_DIR="${1:?Usage: ./corrupt-checkpoint.sh <checkpoint-dir>}"

LATEST=$(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$LATEST" ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

STATE_FILE="$LATEST/state.pt"

if [ ! -f "$STATE_FILE" ]; then
    echo "No state.pt in $LATEST"
    exit 1
fi

SIZE=$(stat -c%s "$STATE_FILE")
echo "=== Fault Injection: Checkpoint Corruption ==="
echo "Target: $STATE_FILE ($SIZE bytes)"
echo "Writing random bytes at offset 1024..."

dd if=/dev/urandom of="$STATE_FILE" bs=1 count=512 seek=1024 conv=notrunc 2>/dev/null

echo "Corrupted. Checksum no longer matches meta.json."
echo ""
echo "Recovery test: restart training — it should skip this checkpoint and use the previous one."
echo "Run: make train"
