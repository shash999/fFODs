"""COCO val2017 mAP for the ONNX model (FP32 reference) or the TensorRT engine (as deployed).

Preprocessing matches configs/pgie_yolo_e2e.txt: RGB, /255, symmetric letterbox with black
padding, bilinear resize (scaling-filter=1). So ONNX vs engine isolates the FP16 conversion.

  # Windows (CPU, a few minutes):
  python scripts/eval_coco.py --backend onnx
  # Container / Jetson (the actual engine):
  python3 scripts/eval_coco.py --backend trt

Each run reports mAP twice: at conf 0.001 (standard, comparable to published numbers)
and at the deployed threshold from the nvinfer config (0.25).
"""

import argparse
import contextlib
import ctypes
import ctypes.util
import io
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
IMGSZ = 640
DEPLOY_CONF = 0.25  # [class-attrs-all] pre-cluster-threshold


def letterbox(img, interp, pad):
    """Symmetric letterbox like nvinfer (maintain-aspect-ratio=1, symmetric-padding=1)."""
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), pad, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas, r, left, top


class OnnxRunner:
    def __init__(self, path):
        import onnxruntime as ort
        # AzureExecutionProvider (default in some wheels) is a remote-call stub; skip it.
        providers = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.name = f"onnxruntime {ort.__version__} ({self.sess.get_providers()[0]})"

    def __call__(self, x):
        return self.sess.run(None, {self.inp: x})[0]


class TrtRunner:
    """Minimal TensorRT 10 runner; CUDA calls go through ctypes so no extra packages are needed."""

    def __init__(self, path):
        import tensorrt as trt
        lib = ctypes.util.find_library("cudart") or "libcudart.so.12"
        self.cuda = ctypes.CDLL(lib)
        self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.max_batch = self.engine.get_tensor_profile_shape(self.inp, 0)[2][0]
        self.d_in = self._malloc(self.max_batch * 3 * IMGSZ * IMGSZ * 4)
        self.d_out = self._malloc(self.max_batch * 300 * 6 * 4)
        self.ctx.set_tensor_address(self.inp, self.d_in.value)
        self.ctx.set_tensor_address(self.out, self.d_out.value)
        self.name = f"TensorRT {trt.__version__} ({Path(path).name})"

    def _malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.cuda.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    @staticmethod
    def _check(err, what):
        if err != 0:
            raise RuntimeError(f"{what} failed with CUDA error {err}")

    def __call__(self, x):
        x = np.ascontiguousarray(x, np.float32)
        self.ctx.set_input_shape(self.inp, x.shape)
        self._check(self.cuda.cudaMemcpy(self.d_in, x.ctypes.data, x.nbytes, 1), "memcpy H2D")
        if not self.ctx.execute_async_v3(0):
            raise RuntimeError("TensorRT execution failed")
        out = np.empty(tuple(self.ctx.get_tensor_shape(self.out)), np.float32)
        self._check(self.cuda.cudaMemcpy(out.ctypes.data, self.d_out, out.nbytes, 2), "memcpy D2H")
        return out


def coco_eval(coco, dets, img_ids):
    if not dets:
        return [0.0] * 12
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["onnx", "trt"], required=True)
    ap.add_argument("--model", help="default: models/model.onnx or models/model.engine")
    ap.add_argument("--data", default=str(ROOT / "datasets" / "coco"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick check)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--interp", choices=["linear", "nearest"], default="linear")
    ap.add_argument("--pad", type=int, default=0, help="letterbox fill: 0 = DeepStream, 114 = Ultralytics")
    ap.add_argument("--results", default=str(ROOT / "eval_results.jsonl"))
    args = ap.parse_args()

    model = Path(args.model or ROOT / "models" / ("model.onnx" if args.backend == "onnx" else "model.engine"))
    runner = OnnxRunner(model) if args.backend == "onnx" else TrtRunner(model)
    interp = cv2.INTER_LINEAR if args.interp == "linear" else cv2.INTER_NEAREST

    data = Path(args.data)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data / "annotations" / "instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())  # model class index -> COCO category id
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]

    print(f"{runner.name} | {len(img_ids)} images | interp={args.interp} pad={args.pad}")
    dets = []
    t_infer = 0.0
    t0 = time.perf_counter()
    for start in range(0, len(img_ids), args.batch):
        ids = img_ids[start: start + args.batch]
        batch, metas = [], []
        for img_id in ids:
            info = coco.imgs[img_id]
            img = cv2.imread(str(data / "val2017" / info["file_name"]))
            padded, r, dx, dy = letterbox(img, interp, args.pad)
            batch.append(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            metas.append((img_id, r, dx, dy, info["width"], info["height"]))
        x = np.stack(batch).astype(np.float32) / 255.0

        ti = time.perf_counter()
        out = runner(x)
        t_infer += time.perf_counter() - ti

        for (img_id, r, dx, dy, w, h), d in zip(metas, out):
            d = d[d[:, 4] >= 0.001]
            x1 = np.clip((d[:, 0] - dx) / r, 0, w)
            y1 = np.clip((d[:, 1] - dy) / r, 0, h)
            x2 = np.clip((d[:, 2] - dx) / r, 0, w)
            y2 = np.clip((d[:, 3] - dy) / r, 0, h)
            for i in range(len(d)):
                dets.append({
                    "image_id": img_id,
                    "category_id": cat_ids[int(d[i, 5])],
                    "bbox": [float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])],
                    "score": float(d[i, 4]),
                })
        done = start + len(ids)
        print(f"\r  {done}/{len(img_ids)} images", end="", flush=True)
    elapsed = time.perf_counter() - t0
    print()

    std = coco_eval(coco, dets, img_ids)
    dep = coco_eval(coco, [d for d in dets if d["score"] >= DEPLOY_CONF], img_ids)

    rows = [
        ("mAP50-95", 0), ("mAP50", 1), ("mAP75", 2),
        ("mAP small", 3), ("mAP medium", 4), ("mAP large", 5), ("AR@100", 8),
    ]
    print(f"\n{'metric':<12}{'conf 0.001':>12}{f'conf {DEPLOY_CONF}':>12}")
    for label, i in rows:
        print(f"{label:<12}{std[i] * 100:>12.1f}{dep[i] * 100:>12.1f}")
    print(f"\n{elapsed:.0f} s total, {1000 * t_infer / len(img_ids):.2f} ms/img inference "
          f"(batch {args.batch}, includes host<->device copies)")

    result = {
        "backend": runner.name, "machine": platform.machine(), "images": len(img_ids),
        "interp": args.interp, "pad": args.pad,
        "map50_95": round(std[0] * 100, 2), "map50": round(std[1] * 100, 2),
        "map50_95_deploy": round(dep[0] * 100, 2), "map50_deploy": round(dep[1] * 100, 2),
    }
    with open(args.results, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"appended to {args.results}")


if __name__ == "__main__":
    main()
