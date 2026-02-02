#!/usr/bin/env bash

# Defaults
STRATEGY="greedy"
BEAM_SIZE=5
MAX_SEQ_LEN=100
SENTENCE="hello"
DEVICE="cpu"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --strategy) STRATEGY="$2"; shift 2 ;;
    --beam_size) BEAM_SIZE="$2"; shift 2 ;;
    --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --sentence) SENTENCE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

# Build command
CMD="python translate.py --strategy $STRATEGY --beam_size $BEAM_SIZE --max_seq_len $MAX_SEQ_LEN --sentence \"$SENTENCE\" --device $DEVICE"

echo $CMD
eval $CMD