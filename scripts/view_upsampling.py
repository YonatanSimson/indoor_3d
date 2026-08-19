"""Compare depth upsampling methods against the RGB that guides them.

Upsampling is judged on one thing: do depth edges land where the image says the
object boundary is? That is visible per pixel and invisible in a fused mesh, so
this puts the source depth, the guide image and each method's output side by
side on the same frame.

    python scripts/view_upsampling.py
    python scripts/view_upsampling.py --scale 4 --frame 120

Left/Right step frames (shift for x10), Z toggles a zoomed crop, arrow-drag on
the RGB panel is not needed -- the crop follows --crop-centre.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample

DEFAULT_SESSION = r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"


class UpsamplingViewer:
    def __init__(self, sequence, index: int, scale: float, min_confidence: int, zoom: bool):
        self.sequence = sequence
        self.index = index
        self.scale = scale
        self.min_confidence = min_confidence
        self.zoom = zoom

        # sharex/sharey ties all six panels together, so zooming or panning any
        # one moves the rest. That only lines up because every image is drawn
        # into the same normalised 0..1 extent below, despite the panels having
        # three different pixel resolutions.
        self.figure, self.axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
        self.figure.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.03,
                                    hspace=0.15, wspace=0.08)
        self.axes = self.axes.ravel()
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw()

    def _on_key(self, event) -> None:
        key = (event.key or "")
        step = 10 if key.startswith("shift+") else 1
        key = key.replace("shift+", "")

        if key == "right":
            self.index = min(self.index + step, len(self.sequence) - 1)
        elif key == "left":
            self.index = max(self.index - step, 0)
        elif key == "z":
            self.zoom = not self.zoom
        else:
            return
        self.draw()

    def draw(self) -> None:
        frame = self.sequence[self.index]
        rgb = self.sequence.load_rgb(frame)
        raw = self.sequence.load_depth(frame)
        confidence = self.sequence.load_confidence(frame)

        cleaned, _ = clean_depth(raw, confidence, min_confidence=self.min_confidence)
        h, w = cleaned.shape
        target_w, target_h = int(w * self.scale), int(h * self.scale)

        outputs = {
            method: upsample(cleaned, rgb, confidence, target_w, target_h, method)
            for method in ("bilinear", "joint", "solver")
        }

        valid = cleaned > 0
        vmin = float(cleaned[valid].min()) if valid.any() else 0.0
        vmax = float(np.percentile(cleaned[valid], 99)) if valid.any() else 5.0

        def show(axis, image, title, **kwargs):
            axis.clear()
            # Every panel spans the same 0..1 box regardless of its resolution,
            # which is what lets the shared axes align them pixel-for-pixel.
            axis.imshow(image, extent=(0.0, 1.0, 1.0, 0.0), aspect="auto", **kwargs)
            axis.set_title(title, fontsize=10)
            axis.set_xticks([]); axis.set_yticks([])

        def crop(image):
            # Zoom is applied through the shared axis limits rather than by
            # slicing, so the toolbar's pan and zoom keep working afterwards.
            return image

        show(self.axes[0], crop(rgb), f"RGB guide  {rgb.shape[1]}x{rgb.shape[0]}")

        show(self.axes[1], crop(np.where(valid, cleaned, np.nan)),
             f"source depth  {w}x{h}  (nearest)", cmap="turbo", vmin=vmin, vmax=vmax,
             interpolation="nearest")

        if confidence is not None:
            counts = np.bincount(confidence.ravel(), minlength=3) / confidence.size
            show(self.axes[2], crop(confidence),
                 f"confidence  low {counts[0]:.0%} / med {counts[1]:.0%} / high {counts[2]:.0%}",
                 cmap=matplotlib.colors.ListedColormap(["#b2182b", "#f6c141", "#1a9850"]),
                 vmin=0, vmax=2, interpolation="nearest")
        else:
            show(self.axes[2], np.zeros((h, w)), "no confidence map", cmap="grey")

        for axis, method in zip(self.axes[3:], ("bilinear", "joint", "solver")):
            out = outputs[method]
            filled = (out > 0).mean()
            show(axis, crop(np.where(out > 0, out, np.nan)),
                 f"{method}  {target_w}x{target_h}  filled {filled:.0%}",
                 cmap="turbo", vmin=vmin, vmax=vmax, interpolation="nearest")

        # Applied once, to a single axis: the shared link propagates it.
        if self.zoom:
            self.axes[0].set_xlim(0.25, 0.75)
            self.axes[0].set_ylim(0.75, 0.25)
        else:
            self.axes[0].set_xlim(0.0, 1.0)
            self.axes[0].set_ylim(1.0, 0.0)

        self.figure.suptitle(
            f"frame {self.index + 1}/{len(self.sequence)}  {frame.folder.name}   "
            f"upsample x{self.scale:g}   "
            f"[{'zoomed' if self.zoom else 'full frame'}]  "
            "-- left/right = frame, shift = x10, z = zoom, toolbar pan/zoom moves all panels",
            fontsize=11,
        )
        self.figure.canvas.draw_idle()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--zoom", action="store_true", help="start zoomed into the centre")
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)
    print(f"{len(sequence)} frames; showing frame {args.frame} at x{args.scale:g}")
    print("left/right = step, shift+arrow = x10, z = toggle zoom")
    UpsamplingViewer(sequence, args.frame, args.scale, args.min_confidence, args.zoom)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
