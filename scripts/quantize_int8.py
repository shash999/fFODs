"""Post-training INT8 quantization with NVIDIA ModelOpt -> models/model.int8.onnx (QDQ).

The calibrated ranges are baked into the ONNX as QuantizeLinear/DequantizeLinear nodes,
so this runs once (here, on the laptop) and the same file builds INT8 engines on any
device: `scripts/build_engine.sh 8 int8` on the laptop, `... 4 int8` on the Jetson.

Calibration images get exactly the deployment preprocessing (eval_coco.letterbox).
They come from COCO val2017; calibration only sets activation ranges (no labels), so
reusing val images has negligible effect on the mAP measured afterwards.

  python scripts/quantize_int8.py              # 512 images, entropy calibration
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from modelopt.onnx.quantization import quantize
from onnxruntime.quantization.calibrate import CalibrationDataReader

from eval_coco import letterbox

ROOT = Path(__file__).resolve().parents[1]


class CocoReader(CalibrationDataReader):
    """Streams preprocessed images one at a time so calibration never holds them all in RAM."""

    def __init__(self, files, input_name):
        self.files = files
        self.input_name = input_name
        self.rewind()

    def _load(self, path):
        padded, *_ = letterbox(cv2.imread(str(path)), cv2.INTER_LINEAR, 0)
        x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        return {self.input_name: np.ascontiguousarray(x, np.float32) / 255.0}

    def get_next(self):
        path = next(self.it, None)
        return None if path is None else self._load(path)

    def get_first(self):  # ModelOpt uses this to probe the graph before calibrating
        return self._load(self.files[0])

    def rewind(self):
        self.it = iter(self.files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=str(ROOT / "models" / "model.onnx"))
    ap.add_argument("--out", default=str(ROOT / "models" / "model.int8.onnx"))
    ap.add_argument("--images", default=str(ROOT / "datasets" / "coco" / "val2017"))
    ap.add_argument("--num", type=int, default=512, help="calibration images")
    ap.add_argument("--method", choices=["entropy", "max"], default="entropy")
    ap.add_argument("--exclude-ops", nargs="*", default=[],
                    help="op types to keep in high precision, e.g. MatMul Softmax")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.images).glob("*.jpg"))
    if not files:
        raise SystemExit(f"no images in {args.images}; run scripts/get_coco_val.py first")
    random.Random(args.seed).shuffle(files)
    files = files[: args.num]

    reader = CocoReader(files, "images")
    print(f"calibrating on {len(files)} images ({args.method}) -> {args.out}")
    quantize(
        onnx_path=args.onnx,
        quantize_mode="int8",
        calibration_method=args.method,
        calibration_data_reader=reader,
        calibration_eps=["cpu"],
        # Non-quantized layers stay FP32 in the ONNX; TensorRT picks FP16 for them with --fp16.
        high_precision_dtype="fp32",
        op_types_to_exclude=args.exclude_ops or None,
        output_path=args.out,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
