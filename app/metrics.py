"""Run-time measurements for app/pipeline.py: per-frame latency and GPU telemetry."""

import subprocess
import threading
import time
from collections import defaultdict

import numpy as np

# nvidia-smi clocks_event_reasons bits (0x1 = idle, not a slowdown).
THROTTLE_BITS = {
    0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
    0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks",
}


def percentiles(values):
    a = np.asarray(values)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"mean": a.mean(), "p50": p50, "p95": p95, "p99": p99, "max": a.max()}


class LatencyTracker:
    """Decode-to-detection latency per frame.

    stamp() is a buffer probe on each decoder output pad and records arrival time keyed by
    (source, PTS). After inference, record() looks up the same frame via NvDsFrameMeta.buf_pts.
    The difference covers batching in nvstreammux, queueing, preprocessing and inference.
    """

    MAX_PENDING = 10_000  # per source; guards against PTS that never come back

    def __init__(self):
        self.pending = defaultdict(dict)
        self.window = []
        self.total = []

    def stamp(self, pad, info, source_id):
        pending = self.pending[source_id]
        if len(pending) > self.MAX_PENDING:
            pending.clear()
        pending[info.get_buffer().pts] = time.perf_counter_ns()
        return 1  # Gst.PadProbeReturn.OK

    def record(self, source_id, pts, now_ns):
        t_in = self.pending[source_id].pop(pts, None)
        if t_in is not None:
            self.window.append((now_ns - t_in) / 1e6)

    def close_window(self):
        if not self.window:
            return None
        stats = percentiles(self.window)
        self.total.extend(self.window)
        self.window = []
        return stats

    def summary(self):
        self.total.extend(self.window)
        self.window = []
        if not self.total:
            return {}
        return {f"latency_ms_{k}": round(float(v), 2) for k, v in percentiles(self.total).items()}


class GpuMonitor:
    """Samples GPU telemetry from a single long-running `nvidia-smi -lms` process (dGPU only)."""

    FIELDS = ["utilization.gpu", "utilization.decoder", "temperature.gpu",
              "power.draw", "clocks.sm", "clocks_event_reasons.active"]

    def __init__(self, period_ms=1000):
        self.samples = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}",
             "--format=csv,noheader,nounits", f"-lms={period_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            try:
                sm, dec, temp, power, clock = (float(p) for p in parts[:5])
                reasons = int(parts[5], 16)
            except (ValueError, IndexError):
                continue
            self.samples.append((sm, dec, temp, power, clock, reasons))

    def last(self):
        return self.samples[-1] if self.samples else None

    def stop(self):
        self.proc.terminate()

    def summary(self, skip_first=5):
        s = self.samples[skip_first:] or self.samples
        if not s:
            return {}
        a = np.array([x[:5] for x in s])
        reasons = 0
        for x in s:
            reasons |= x[5]
        return {
            "gpu_util_avg": round(float(a[:, 0].mean()), 1),
            "decoder_util_avg": round(float(a[:, 1].mean()), 1),
            "gpu_temp_max": int(a[:, 2].max()),
            "gpu_power_avg_w": round(float(a[:, 3].mean()), 1),
            "gpu_clock_avg_mhz": int(a[:, 4].mean()),
            "gpu_clock_min_mhz": int(a[:, 4].min()),
            "throttle_reasons": [n for bit, n in THROTTLE_BITS.items() if reasons & bit] or "none",
        }
