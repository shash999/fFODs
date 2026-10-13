# Laptop (x86 + RTX 4060 via WSL2/Docker Desktop) dev image: DeepStream 7.1 + Python bindings.
# Matches the DeepStream version on JetPack 6.1/6.2, so configs and code are identical on both.
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-pip python3-dev python3-gi python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/*

ARG PYDS_VER=1.2.0
RUN pip3 install --no-cache-dir \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v${PYDS_VER}/pyds-${PYDS_VER}-cp310-cp310-linux_x86_64.whl

# Accuracy evaluation (scripts/eval_coco.py). numpy<2 keeps pyds happy.
RUN pip3 install --no-cache-dir "numpy<2" "opencv-python-headless<4.11" pycocotools \
    && (python3 -c "import tensorrt" 2>/dev/null || pip3 install --no-cache-dir "tensorrt-cu12==10.3.*")

WORKDIR /work
