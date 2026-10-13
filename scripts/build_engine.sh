#!/usr/bin/env bash
# Build a TensorRT engine from models/model.onnx ON THE TARGET DEVICE.
# Engines are tied to the GPU + TensorRT version: build separately on laptop and Jetson.
#
# Usage: scripts/build_engine.sh [max_batch] [precision: fp16|int8|fp32]
# Takes ~1-3 min on the RTX 4060, ~5-15 min on Orin Nano.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAX_BATCH="${1:-4}"
PREC="${2:-fp16}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
INPUT="images"   # Ultralytics input name
IMGSZ=640

PREC_FLAGS=()
case "$PREC" in
  fp16) ONNX="$ROOT/models/model.onnx";      ENGINE="$ROOT/models/model.engine";      PREC_FLAGS=(--fp16) ;;
  fp32) ONNX="$ROOT/models/model.onnx";      ENGINE="$ROOT/models/model.fp32.engine" ;;
  # QDQ model from scripts/quantize_int8.py: INT8 where calibrated, FP16 elsewhere.
  int8) ONNX="$ROOT/models/model.int8.onnx"; ENGINE="$ROOT/models/model.int8.engine"; PREC_FLAGS=(--int8 --fp16) ;;
  *) echo "unknown precision $PREC"; exit 1 ;;
esac

[[ -f "$ONNX" ]] || { echo "missing $ONNX - run scripts/export_onnx.py (and quantize_int8.py for int8) first"; exit 1; }

"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  "${PREC_FLAGS[@]}" \
  --minShapes="$INPUT:1x3x${IMGSZ}x${IMGSZ}" \
  --optShapes="$INPUT:${MAX_BATCH}x3x${IMGSZ}x${IMGSZ}" \
  --maxShapes="$INPUT:${MAX_BATCH}x3x${IMGSZ}x${IMGSZ}" \
  --shapes="$INPUT:${MAX_BATCH}x3x${IMGSZ}x${IMGSZ}" \
  --memPoolSize=workspace:2048 \
  --useCudaGraph \
  2>&1 | tee "$ROOT/models/build_${PREC}_b${MAX_BATCH}.log" | grep -E "Throughput|Latency|GPU Compute Time|PASSED|FAILED|Error"

echo "engine -> $ENGINE  (Throughput x $MAX_BATCH = images/sec at batch $MAX_BATCH)"
