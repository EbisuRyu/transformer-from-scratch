#!/usr/bin/env bash

# Defaults
SRC_FILE="./data/iwslt2015_en_vi/test.en"
REF_FILE="./data/iwslt2015_en_vi/test.vi"
CHECKPOINT=""
DEVICE="cpu"
BERT_MODEL="xlm-roberta-base"
MAX_SEQ_LEN=128
STRATEGY="greedy"
BEAM_SIZE=4

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --src-file) SRC_FILE="$2"; shift 2 ;;
    --ref-file) REF_FILE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --bert-model) BERT_MODEL="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --beam-size) BEAM_SIZE="$2"; shift 2 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

# Ensure checkpoint is provided
if [[ -z "$CHECKPOINT" ]]; then
  echo "Error: --checkpoint is required"
  exit 1
fi

# Build command
CMD="python evaluate.py \
  --src-file \"$SRC_FILE\" \
  --ref-file \"$REF_FILE\" \
  --checkpoint \"$CHECKPOINT\" \
  --device $DEVICE \
  --bert-model $BERT_MODEL \
  --max-seq-len $MAX_SEQ_LEN \
  --strategy $STRATEGY \
  --beam-size $BEAM_SIZE"

echo $CMD
eval $CMD
