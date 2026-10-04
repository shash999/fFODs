# fFODs — fast object detection pipeline

DeepStream 7.1 + TensorRT. An end-to-end YOLO model (NMS inside the engine) is run by
`nvinfer`. The same code and configs run on the laptop (RTX 4060, via Docker/WSL2) and on
the Jetson Orin Nano.

```
nvurisrcbin x N ─► nvstreammux ─► queue ─► nvinfer (TRT engine) ─► queue ─► [tiler ─► nvdsosd] ─► sink
   (NVDEC)          (batch N)                 + e2e parser .so                    (fake | display | file)
```

## Layout
| Path | What |
|---|---|
| `scripts/export_onnx.py` | YOLO `.pt` → `models/model.onnx` (dynamic batch, `[B,N,6]` output) |
| `scripts/build_engine.sh` | `trtexec` → `models/model.engine` (**per device**) + throughput numbers |
| `parser/` | Tiny `nvinfer` bbox parser for the `[N,6]` output |
| `configs/pgie_yolo_e2e.txt` | `nvinfer` config |
| `app/pipeline.py` | Multi-stream pipeline with an FPS readout |
| `docker/Dockerfile` | Laptop DeepStream 7.1 image with `pyds` |

## 1. Export ONNX (laptop, once)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install ultralytics onnx onnxslim
python scripts/export_onnx.py --model yolo26n.pt --batch 8
```

## 2. Laptop (RTX 4060)
```powershell
docker build -t ffods-ds docker        # big pull (~10+ GB) the first time
.\scripts\laptop_shell.ps1
```
Inside the container:
```bash
make -C parser
bash scripts/build_engine.sh 8 fp16
python3 app/pipeline.py -n 8 --loop                    # benchmark: FPS printed every 2 s
python3 app/pipeline.py -n 4 --sink file -o out.mp4    # Ctrl-C to finish the file
```

## 3. Jetson Orin Nano
Copy the project folder (including `models/model.onnx`), then:
```bash
bash scripts/setup_jetson.sh
sudo nvpmodel -m <MAXN id>; sudo jetson_clocks
bash scripts/build_engine.sh 4 fp16
python3 app/pipeline.py -n 4 --loop
```

## FPS levers (biggest first)
1. **Model size** — `yolo26n` vs `s`/`m`, and input size (`--imgsz 480/416` at export).
2. **Precision** — FP16 is the default; INT8 needs a calibration cache to keep accuracy.
3. **Batching** — more streams per batch means better GPU use; keep the engine's max batch ≥ `-n`.
4. **`--interval K`** — infer every K+1 frames. Add `nvtracker` to fill the skipped frames.
5. **Output** — `fake` sink is the true ceiling. The tiler, overlay and encoder cost FPS.
   Orin Nano encodes on the CPU (no NVENC).
6. **Mux resolution** — set `--mux-width/height` to the source resolution so frames aren't scaled twice.
