"""Sanity-check models/model.onnx on an image with ONNX Runtime (no DeepStream/GPU needed).

Uses the same preprocessing as configs/pgie_yolo_e2e.txt (RGB, /255, centered letterbox),
so if boxes look right here, the parser + nvinfer config should agree.

  python scripts/test_onnx.py                  # Ultralytics' bundled bus.jpg
  python scripts/test_onnx.py --image my.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--model", default=str(ROOT / "models" / "model.onnx"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "test_out.jpg"))
    args = ap.parse_args()

    if args.image is None:
        import ultralytics
        args.image = str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg")
    labels = (ROOT / "configs" / "labels.txt").read_text().splitlines()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = 640

    img = cv2.imread(args.image)
    padded, r, dx, dy = letterbox(img, size)
    x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    x = np.repeat(x, 2, axis=0)  # batch of 2 to prove the dynamic batch dim works

    out = sess.run(None, {inp.name: x})[0]
    print(f"input {inp.name} {list(x.shape)} -> output {list(out.shape)}")
    assert out.ndim == 3 and out.shape[2] == 6, "expected [B, N, 6]"
    assert np.allclose(out[0], out[1], atol=1e-4), "batch items differ for identical input"

    dets = out[0][out[0, :, 4] >= args.conf]
    for x1, y1, x2, y2, score, cls in dets:
        # undo letterbox -> original image pixels
        x1, x2 = (x1 - dx) / r, (x2 - dx) / r
        y1, y2 = (y1 - dy) / r, (y2 - dy) / r
        name = labels[int(cls)]
        print(f"  {name:12s} {score:.2f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {score:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.out, img)
    print(f"{len(dets)} detections -> {args.out}")


if __name__ == "__main__":
    main()
