#!/usr/bin/env bash
set -euo pipefail

SCRIPT=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"
nohup bash "$SCRIPT" > "$OUTPUT_DIR/train.log" 2>&1 &
echo "Training started, PID: $!"
echo "$!" > "$OUTPUT_DIR/train.pid"
