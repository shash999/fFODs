"""High-FPS multi-stream detection pipeline (DeepStream 7.1, nvinfer + TensorRT).

  nvurisrcbin x N -> nvstreammux -> queue -> nvinfer -> queue -> [tiler -> osd] -> sink

Every stage after the decoder stays in GPU (NVMM) memory. Queues put decode, inference
and rendering on separate threads so they overlap instead of running serially.

Examples:
  python3 app/pipeline.py                                   # sample clip, benchmark (fakesink)
  python3 app/pipeline.py -n 4 --loop                       # 4 copies of the sample, looping
  python3 app/pipeline.py -i rtsp://cam1/stream -i rtsp://cam2/stream --sink display
  python3 app/pipeline.py -i file:///work/video.mp4 --sink file -o out.mp4
  python3 app/pipeline.py -n 8 --loop --duration 600 --latency --gpu-stats   # soak test
"""

import argparse
import csv
import json
import platform
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from metrics import GpuMonitor, LatencyTracker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URI = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
IS_JETSON = platform.machine() == "aarch64"


def jetson_has_nvenc():
    # Orin Nano has NVDEC but no NVENC.
    try:
        return "Orin Nano" not in Path("/proc/device-tree/model").read_text()
    except OSError:
        return True


def device_name():
    if IS_JETSON:
        return Path("/proc/device-tree/model").read_text().strip("\x00\n")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout
        return out.splitlines()[0].strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        return platform.node()


def silence_loop_warnings():
    """nvurisrcbin's file-loop restarts push data before the new segment event, so the
    decoders emit a harmless 'Got data flow before segment event' g_warning per loop.
    Drop just that message; every other GStreamer warning still prints."""
    def handler(domain, level, message, *_):
        if "before segment event" not in message:
            print(f"({domain}-WARNING): {message}", file=sys.stderr)

    GLib.log_set_handler("GStreamer", GLib.LogLevelFlags.LEVEL_WARNING, handler, None)


def make(factory, name=None, **props):
    el = Gst.ElementFactory.make(factory, name)
    if el is None:
        raise RuntimeError(f"Could not create GStreamer element '{factory}'")
    for k, v in props.items():
        el.set_property(k.replace("_", "-"), v)
    return el


class FpsProbe:
    """Counts frames per source after inference, prints FPS every `period` seconds,
    and keeps run totals for summary(). Timing starts at the first batch, so engine
    loading and pipeline startup are not counted."""

    def __init__(self, num_sources, period=2.0, print_dets=False, latency=None, gpu=None):
        self.num_sources = num_sources
        self.period = period
        self.print_dets = print_dets
        self.latency = latency
        self.gpu = gpu
        self.timeseries = None  # csv.writer, set by enable_timeseries()
        self.frames = defaultdict(int)
        self.objects = 0
        self.t0 = None
        self.t_start = None
        self.windows = []  # total FPS of each completed window
        self.total_frames = defaultdict(int)
        self.total_objects = 0

    def __call__(self, pad, info):
        now_ns = time.perf_counter_ns()
        now = now_ns / 1e9
        if self.t0 is None:
            self.t0 = self.t_start = now
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame = pyds.NvDsFrameMeta.cast(l_frame.data)
            self.frames[frame.pad_index] += 1
            self.objects += frame.num_obj_meta
            if self.latency:
                self.latency.record(frame.pad_index, frame.buf_pts, now_ns)
            if self.print_dets:
                self._print_objects(frame)
            l_frame = l_frame.next

        dt = now - self.t0
        if dt >= self.period:
            self._close_window(dt)
        return Gst.PadProbeReturn.OK

    TIMESERIES_HEADER = ["t_s", "fps_total", "objs_per_frame", "latency_p50_ms", "latency_p99_ms",
                         "gpu_util", "decoder_util", "temp_c", "power_w", "clock_mhz", "throttle"]

    def enable_timeseries(self, path):
        self._ts_file = open(path, "w", newline="", buffering=1)  # line-buffered: survives crashes
        self.timeseries = csv.writer(self._ts_file)
        self.timeseries.writerow(self.TIMESERIES_HEADER)

    def _close_window(self, dt):
        per_stream = [self.frames[i] / dt for i in range(self.num_sources)]
        total = sum(per_stream)
        n = sum(self.frames.values())
        avg_obj = self.objects / max(1, n)
        streams = " ".join(f"{f:6.1f}" for f in per_stream)
        line = f"FPS total {total:7.1f} | per stream [{streams}] | objs/frame {avg_obj:4.1f}"
        lat = self.latency.close_window() if self.latency else None
        if lat:
            line += f" | latency p50 {lat['p50']:5.1f} p99 {lat['p99']:5.1f} ms"
        g = self.gpu.last() if self.gpu else None
        if g:
            line += f" | gpu {g[0]:3.0f}% dec {g[1]:3.0f}% {g[2]:.0f}C {g[3]:.0f}W {g[4]:.0f}MHz"
        print(line)
        if self.timeseries:
            self.timeseries.writerow([
                round(time.perf_counter() - self.t_start, 1), round(total, 1), round(avg_obj, 2),
                *(round(float(lat[k]), 2) if lat else "" for k in ("p50", "p99")),
                *(g[:5] if g else [""] * 5), hex(g[5]) if g else "",
            ])
        self.windows.append(total)
        for i, f in self.frames.items():
            self.total_frames[i] += f
        self.total_objects += self.objects
        self.frames.clear()
        self.objects = 0
        self.t0 = time.perf_counter()

    def summary(self, args):
        if self.t_start is None:
            return None
        # Fold in the partial last window (not printed, not used for min/max).
        for i, f in self.frames.items():
            self.total_frames[i] += f
        self.total_objects += self.objects
        self.frames.clear()
        self.objects = 0

        elapsed = time.perf_counter() - self.t_start
        frames = sum(self.total_frames.values())
        steady = self.windows[1:] or self.windows  # first window includes warm-up
        stats = {
            "device": device_name(),
            "streams": self.num_sources,
            "sink": args.sink,
            "inference": not args.no_infer,
            "engine": Path(args.engine).name if args.engine else "config default",
            "interval": args.interval,
            "seconds": round(elapsed, 1),
            "frames": frames,
            "avg_fps_total": round(frames / elapsed, 1),
            "avg_fps_per_stream": round(frames / elapsed / self.num_sources, 1),
            "min_fps_total": round(min(steady), 1) if steady else None,
            "max_fps_total": round(max(steady), 1) if steady else None,
            "avg_ms_per_frame": round(1000 * elapsed / max(1, frames), 3),
            "avg_objects_per_frame": round(self.total_objects / max(1, frames), 2),
            "realtime_30fps_streams": int(frames / elapsed // 30),
        }
        # Throttling check: first vs last minute of steady state (windows are `period` s).
        per_min = int(60 / self.period)
        if len(steady) >= 3 * per_min:
            stats["fps_first_minute"] = round(sum(steady[:per_min]) / per_min, 1)
            stats["fps_last_minute"] = round(sum(steady[-per_min:]) / per_min, 1)
        if self.latency:
            stats.update(self.latency.summary())
        if self.gpu:
            stats.update(self.gpu.summary())
        width = max(len(k) for k in stats)
        print("\n===== run summary =====")
        for k, v in stats.items():
            print(f"  {k:<{width}}  {v}")
        return stats

    @staticmethod
    def _print_objects(frame):
        l_obj = frame.obj_meta_list
        while l_obj is not None:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            r = obj.rect_params
            print(f"src{frame.pad_index} f{frame.frame_num} {obj.obj_label} {obj.confidence:.2f} "
                  f"[{r.left:.0f},{r.top:.0f},{r.width:.0f},{r.height:.0f}]")
            l_obj = l_obj.next


def add_sources(pipeline, streammux, uris, loop, latency=None):
    for i, uri in enumerate(uris):
        src = make("nvurisrcbin", f"src-{i}", uri=uri)
        if uri.startswith("file://"):
            src.set_property("file-loop", loop)
        else:
            src.set_property("rtsp-reconnect-interval", 5)
        sinkpad = streammux.request_pad_simple(f"sink_{i}")

        def on_pad_added(_bin, pad, sinkpad=sinkpad, i=i):
            caps = pad.get_current_caps() or pad.query_caps(None)
            if not caps.get_structure(0).get_name().startswith("video"):
                return
            if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
                print(f"source {i}: failed to link to streammux", file=sys.stderr)
            if latency:
                pad.add_probe(Gst.PadProbeType.BUFFER, latency.stamp, i)

        src.connect("pad-added", on_pad_added)
        pipeline.add(src)


def build_output(pipeline, args, num_sources):
    """Return the list of elements after inference, ending with the sink."""
    if args.sink == "fake":
        return [make("fakesink", "sink", sync=False, enable_last_sample=False)]

    cols = int(num_sources ** 0.5 + 0.999)
    rows = (num_sources + cols - 1) // cols
    chain = [
        make("nvmultistreamtiler", "tiler", rows=rows, columns=cols,
             width=args.out_width, height=args.out_height),
        make("nvvideoconvert", "osd-conv"),
        make("nvdsosd", "osd", process_mode=1),  # 1 = GPU
    ]

    if args.sink == "display":
        sink = "nv3dsink" if IS_JETSON else "nveglglessink"
        return chain + [make(sink, "sink", sync=args.sync)]

    # sink == "file"
    if IS_JETSON and not jetson_has_nvenc():
        chain += [
            make("nvvideoconvert", "enc-conv"),
            make("capsfilter", caps=Gst.Caps.from_string("video/x-raw,format=I420")),
            make("x264enc", "encoder", speed_preset="ultrafast", tune="zerolatency",
                 bitrate=args.bitrate // 1000),
        ]
    else:
        chain += [
            make("nvvideoconvert", "enc-conv"),
            make("capsfilter", caps=Gst.Caps.from_string("video/x-raw(memory:NVMM),format=I420")),
            make("nvv4l2h264enc", "encoder", bitrate=args.bitrate),
        ]
    return chain + [
        make("h264parse"),
        make("qtmux"),
        make("filesink", "sink", location=args.output, sync=False),
    ]


def build_pipeline(args):
    uris = args.input or [DEFAULT_URI]
    if args.num_sources > len(uris):
        uris = (uris * args.num_sources)[: args.num_sources]
    n = len(uris)
    live = any(u.startswith(("rtsp://", "rtmp://", "http")) for u in uris)

    pipeline = Gst.Pipeline.new("ffods")
    streammux = make(
        "nvstreammux", "mux",
        batch_size=n, width=args.mux_width, height=args.mux_height,
        live_source=live,
        # Max wait to fill a batch: about one frame interval at the source rate.
        batched_push_timeout=args.push_timeout_us,
    )
    if args.no_infer:
        # Decode + mux only: measures the video-decode ceiling for comparison.
        infer = [make("identity", "pgie")]
    else:
        pgie = make("nvinfer", "pgie", config_file_path=str(ROOT / args.config), batch_size=n)
        if args.engine:
            pgie.set_property("model-engine-file", str((ROOT / args.engine).resolve()))
        if args.interval:
            pgie.set_property("interval", args.interval)
        infer = [make("queue", "q-infer"), pgie]

    elements = [
        streammux,
        *infer,
        make("queue", "q-out"),
        *build_output(pipeline, args, n),
    ]
    for el in elements:
        pipeline.add(el)
    for a, b in zip(elements, elements[1:]):
        if not a.link(b):
            raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

    latency = LatencyTracker() if args.latency else None
    gpu = None
    if args.gpu_stats:
        if IS_JETSON:
            print("--gpu-stats uses nvidia-smi (dGPU only); on Jetson run `tegrastats` alongside.")
        else:
            gpu = GpuMonitor()

    add_sources(pipeline, streammux, uris, args.loop, latency)

    probe = FpsProbe(n, print_dets=args.print_dets, latency=latency, gpu=gpu)
    if args.timeseries:
        probe.enable_timeseries(args.timeseries)
    pipeline.get_by_name("pgie").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)
    return pipeline, probe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", action="append", help="URI (file:///abs/path.mp4 or rtsp://...). Repeatable.")
    ap.add_argument("-n", "--num-sources", type=int, default=1, help="Replicate inputs up to N streams")
    ap.add_argument("--sink", choices=["fake", "display", "file"], default="fake")
    ap.add_argument("-o", "--output", default="out.mp4")
    ap.add_argument("--config", default="configs/pgie_yolo_e2e.txt")
    ap.add_argument("--engine", help="override the config's engine, e.g. models/model.int8.engine")
    ap.add_argument("--loop", action="store_true", help="Loop file sources forever")
    ap.add_argument("--sync", action="store_true", help="Display at real-time speed instead of max speed")
    ap.add_argument("--interval", type=int, default=0, help="Skip N frames between inferences")
    ap.add_argument("--mux-width", type=int, default=1920)
    ap.add_argument("--mux-height", type=int, default=1080)
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--out-height", type=int, default=720)
    ap.add_argument("--push-timeout-us", type=int, default=33000)
    ap.add_argument("--bitrate", type=int, default=4_000_000)
    ap.add_argument("--print-dets", action="store_true", help="Print every detection (slows things down)")
    ap.add_argument("--no-infer", action="store_true", help="Skip inference: measure decode-only FPS")
    ap.add_argument("--stats-json", help="Append the run summary as a JSON line to this file")
    ap.add_argument("--latency", action="store_true", help="Measure decode-to-detection latency per frame")
    ap.add_argument("--gpu-stats", action="store_true", help="Sample GPU/decoder utilization, temp, power, clocks")
    ap.add_argument("--duration", type=int, default=0, help="Stop after N seconds (use with --loop)")
    ap.add_argument("--timeseries", help="Write one CSV row per 2 s window (plot with scripts/plot_timeseries.py)")
    args = ap.parse_args()

    Gst.init(None)
    silence_loop_warnings()
    pipeline, probe = build_pipeline(args)
    n = probe.num_sources
    loop = GLib.MainLoop()

    def on_message(_bus, msg):
        if msg.type == Gst.MessageType.EOS:
            print("EOS")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"ERROR from {msg.src.get_name()}: {err}\n{dbg}", file=sys.stderr)
            loop.quit()
        elif msg.type == Gst.MessageType.WARNING:
            err, _ = msg.parse_warning()
            print(f"WARNING from {msg.src.get_name()}: {err}", file=sys.stderr)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    # Stop = send EOS so MP4 files get finalized. If EOS doesn't arrive within 5 s
    # (or on a second Ctrl-C), quit anyway.
    def stop(reason):
        if getattr(stop, "fired", False):
            loop.quit()
            return GLib.SOURCE_REMOVE
        stop.fired = True
        print(f"\n{reason}: stopping (sending EOS)...")
        pipeline.send_event(Gst.Event.new_eos())
        GLib.timeout_add_seconds(5, lambda: loop.quit() or GLib.SOURCE_REMOVE)
        return GLib.SOURCE_REMOVE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT,
                         lambda *_: stop("Ctrl-C") or GLib.SOURCE_CONTINUE)
    if args.duration:
        GLib.timeout_add_seconds(args.duration, stop, f"{args.duration} s elapsed")

    print(f"Running {n} stream(s), sink={args.sink}. First run may build the TensorRT engine.")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        if probe.gpu:
            probe.gpu.stop()
        stats = probe.summary(args)
        if stats and args.stats_json:
            with open(args.stats_json, "a") as f:
                f.write(json.dumps(stats) + "\n")
            print(f"  appended to {args.stats_json}")


if __name__ == "__main__":
    main()
