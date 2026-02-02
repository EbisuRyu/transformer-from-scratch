#!/usr/bin/env bash

# Defaults
DEVICE="cuda"
BATCH_SIZE=""
EPOCHS=""
RESUME=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

# Build command
CMD="python train.py --device $DEVICE"
[[ -n "$BATCH_SIZE" ]] && CMD="$CMD --batch-size $BATCH_SIZE"
[[ -n "$EPOCHS" ]] && CMD="$CMD --epochs $EPOCHS"
[[ -n "$RESUME" ]] && CMD="$CMD --resume $RESUME"

echo $CMD
eval $CMD