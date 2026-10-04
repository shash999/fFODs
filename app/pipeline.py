"""High-FPS multi-stream detection pipeline (DeepStream 7.1, nvinfer + TensorRT).

  nvurisrcbin x N -> nvstreammux -> queue -> nvinfer -> queue -> [tiler -> osd] -> sink

Every stage after the decoder stays in GPU (NVMM) memory. Queues put decode, inference
and rendering on separate threads so they overlap instead of running serially.

Examples:
  python3 app/pipeline.py                                   # sample clip, benchmark (fakesink)
  python3 app/pipeline.py -n 4 --loop                       # 4 copies of the sample, looping
  python3 app/pipeline.py -i rtsp://cam1/stream -i rtsp://cam2/stream --sink display
  python3 app/pipeline.py -i file:///work/video.mp4 --sink file -o out.mp4
"""

import argparse
import platform
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URI = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
IS_JETSON = platform.machine() == "aarch64"


def jetson_has_nvenc():
    # Orin Nano has NVDEC but no NVENC.
    try:
        return "Orin Nano" not in Path("/proc/device-tree/model").read_text()
    except OSError:
        return True


def make(factory, name=None, **props):
    el = Gst.ElementFactory.make(factory, name)
    if el is None:
        raise RuntimeError(f"Could not create GStreamer element '{factory}'")
    for k, v in props.items():
        el.set_property(k.replace("_", "-"), v)
    return el


class FpsProbe:
    """Counts frames per source after inference and prints FPS every `period` seconds."""

    def __init__(self, num_sources, period=2.0, print_dets=False):
        self.num_sources = num_sources
        self.period = period
        self.print_dets = print_dets
        self.frames = defaultdict(int)
        self.objects = 0
        self.t0 = time.perf_counter()

    def __call__(self, pad, info):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame = pyds.NvDsFrameMeta.cast(l_frame.data)
            self.frames[frame.pad_index] += 1
            self.objects += frame.num_obj_meta
            if self.print_dets:
                self._print_objects(frame)
            l_frame = l_frame.next

        dt = time.perf_counter() - self.t0
        if dt >= self.period:
            per_stream = [self.frames[i] / dt for i in range(self.num_sources)]
            total = sum(per_stream)
            avg_obj = self.objects / max(1, sum(self.frames.values()))
            streams = " ".join(f"{f:6.1f}" for f in per_stream)
            print(f"FPS total {total:7.1f} | per stream [{streams}] | avg objs/frame {avg_obj:.1f}")
            self.frames.clear()
            self.objects = 0
            self.t0 = time.perf_counter()
        return Gst.PadProbeReturn.OK

    @staticmethod
    def _print_objects(frame):
        l_obj = frame.obj_meta_list
        while l_obj is not None:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            r = obj.rect_params
            print(f"src{frame.pad_index} f{frame.frame_num} {obj.obj_label} {obj.confidence:.2f} "
                  f"[{r.left:.0f},{r.top:.0f},{r.width:.0f},{r.height:.0f}]")
            l_obj = l_obj.next


def add_sources(pipeline, streammux, uris, loop):
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
    pgie = make("nvinfer", "pgie", config_file_path=str(ROOT / args.config), batch_size=n)
    if args.interval:
        pgie.set_property("interval", args.interval)

    elements = [
        streammux,
        make("queue", "q-infer"),
        pgie,
        make("queue", "q-out"),
        *build_output(pipeline, args, n),
    ]
    for el in elements:
        pipeline.add(el)
    for a, b in zip(elements, elements[1:]):
        if not a.link(b):
            raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

    add_sources(pipeline, streammux, uris, args.loop)

    probe = FpsProbe(n, print_dets=args.print_dets)
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)
    return pipeline, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", action="append", help="URI (file:///abs/path.mp4 or rtsp://...). Repeatable.")
    ap.add_argument("-n", "--num-sources", type=int, default=1, help="Replicate inputs up to N streams")
    ap.add_argument("--sink", choices=["fake", "display", "file"], default="fake")
    ap.add_argument("-o", "--output", default="out.mp4")
    ap.add_argument("--config", default="configs/pgie_yolo_e2e.txt")
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
    args = ap.parse_args()

    Gst.init(None)
    pipeline, n = build_pipeline(args)
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

    # Ctrl-C sends EOS so MP4 files get finalized; a second Ctrl-C exits immediately.
    def on_sigint(*_):
        if getattr(on_sigint, "fired", False):
            loop.quit()
        on_sigint.fired = True
        print("\nStopping (sending EOS)...")
        pipeline.send_event(Gst.Event.new_eos())
        return GLib.SOURCE_CONTINUE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, on_sigint)

    print(f"Running {n} stream(s), sink={args.sink}. First run may build the TensorRT engine.")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
