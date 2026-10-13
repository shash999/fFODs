"""Plot a --timeseries CSV from app/pipeline.py as small multiples on a shared time axis.

  python scripts/plot_timeseries.py soak.csv              # -> soak.png
  python scripts/plot_timeseries.py soak.csv -o run.png --title "RTX 4060, 8 streams"

Shaded bands mark windows where the GPU reported thermal slowdown.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2
THROTTLE_FILL = "#d9d8d3"
THERMAL_BITS = 0x20 | 0x40  # sw_thermal | hw_thermal


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no data rows")

    def col(name):
        out = []
        for r in rows:
            v = r.get(name, "")
            out.append(float(v) if v not in ("", None) else None)
        return out

    data = {k: col(k) for k in rows[0] if k != "throttle"}
    short = data["t_s"][-1] < 180
    data["t_unit"] = "seconds" if short else "minutes"
    data["t"] = data["t_s"] if short else [t / 60 for t in data["t_s"]]
    data["thermal"] = [bool(r["throttle"]) and int(r["throttle"], 16) & THERMAL_BITS != 0 for r in rows]
    return data


def has(data, key):
    return any(v is not None for v in data.get(key, []))


def shade_thermal(ax, t, thermal):
    start = None
    for i, on in enumerate(thermal + [False]):
        if on and start is None:
            start = t[i - 1] if i else t[0]
        elif not on and start is not None:
            ax.axvspan(start, t[i - 1], color=THROTTLE_FILL, lw=0, zorder=0)
            start = None


def style(ax, title, unit):
    ax.set_facecolor(SURFACE)
    ax.set_title(f"{title}  ({unit})", loc="left", fontsize=10, color=TEXT, pad=6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)


def line(ax, t, y, color, label=None):
    pts = [(a, b) for a, b in zip(t, y) if b is not None]
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=1.5, label=label, solid_capstyle="round")
    if label:  # direct label at the line end, in text ink
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT_2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    d = load(args.csv)
    t = d["t"]

    panels = [("Throughput", "FPS, all streams", [("fps_total", None)])]
    if has(d, "latency_p50_ms"):
        panels.append(("Decode-to-detection latency", "ms",
                       [("latency_p50_ms", "p50"), ("latency_p99_ms", "p99")]))
    if has(d, "temp_c"):
        panels += [
            ("GPU temperature", "°C", [("temp_c", None)]),
            ("SM clock", "MHz", [("clock_mhz", None)]),
            ("Board power", "W", [("power_w", None)]),
            ("Utilization", "%", [("gpu_util", "GPU"), ("decoder_util", "decoder")]),
        ]

    ncols = 2 if len(panels) > 2 else 1
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 2.4 * nrows),
                             sharex=True, squeeze=False, facecolor=SURFACE)
    for ax, (title, unit, series) in zip(axes.flat, panels):
        style(ax, title, unit)
        shade_thermal(ax, t, d["thermal"])
        for i, (key, label) in enumerate(series):
            line(ax, t, d[key], SERIES[i], label)
        ax.margins(x=0.06 if len(series) > 1 else 0.01)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(d["t_unit"], color=TEXT_2, fontsize=9)

    handles = [Patch(color=THROTTLE_FILL, label="thermal throttling")] if any(d["thermal"]) else []
    if handles:
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT_2)
    if args.title:
        fig.suptitle(args.title, x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if (args.title or handles) else 1))

    out = Path(args.out or Path(args.csv).with_suffix(".png"))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
