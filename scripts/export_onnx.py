"""Export an Ultralytics YOLO model to an end-to-end ONNX (NMS inside the graph).

Output tensor: [batch, max_det, 6] = x1, y1, x2, y2, score, class_id in network
pixel coordinates. This is what parser/nvdsparsebbox_e2e.cpp expects.

Run once on the laptop; copy models/model.onnx to the Jetson.
The TensorRT engine must be built separately on each device (scripts/build_engine.sh).
"""

import argparse
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO

# Models whose head is already NMS-free; everything else gets NMS baked in at export.
END2END_FAMILIES = ("yolo26", "yolov10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt", help="Ultralytics weights (auto-downloaded)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="max batch the ONNX is exported for")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "models" / "model.onnx"))
    args = ap.parse_args()

    end2end = any(f in Path(args.model).stem.lower() for f in END2END_FAMILIES)
    # Ultralytics drops the end2end head with dynamic=True unless asked explicitly.
    extra = {"end2end": True} if end2end else {"nms": True}

    exported = YOLO(args.model).export(
        format="onnx", imgsz=args.imgsz, dynamic=True, batch=args.batch, simplify=True, **extra
    )

    graph = onnx.load(exported).graph
    if len(graph.output) != 1:
        raise SystemExit(f"Expected 1 output, got {[o.name for o in graph.output]}")
    dims = graph.output[0].type.tensor_type.shape.dim
    if dims[-1].dim_value != 6:
        raise SystemExit(f"Output is not [B, N, 6] end-to-end detections: {dims}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)
    print(f"input={graph.input[0].name} output={graph.output[0].name} -> {out}")


if __name__ == "__main__":
    main()
