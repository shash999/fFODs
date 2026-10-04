#!/usr/bin/env bash
# One-time setup on the Jetson Orin Nano (JetPack 6.1/6.2 -> DeepStream 7.1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d /opt/nvidia/deepstream/deepstream ]]; then
  sudo apt-get update
  sudo apt-get install -y deepstream-7.1
fi

sudo apt-get install -y build-essential python3-pip python3-dev python3-gi python3-gst-1.0 \
  gstreamer1.0-plugins-ugly   # x264enc: Orin Nano has no hardware encoder

PYDS_VER=1.2.0
python3 -c "import pyds" 2>/dev/null || pip3 install \
  "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v${PYDS_VER}/pyds-${PYDS_VER}-cp310-cp310-linux_aarch64.whl"

make -C "$ROOT/parser"

echo
echo "Power mode: pick the highest (MAXN / MAXN_SUPER) for benchmarks:"
sudo nvpmodel -q || true
echo "  sudo nvpmodel -m <id-of-MAXN>   &&   sudo jetson_clocks"
