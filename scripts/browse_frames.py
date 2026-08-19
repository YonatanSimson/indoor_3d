"""Page through the raw frames of a capture: RGB, depth, confidence, cleaned depth.

Shows what fusion actually consumes, which is the quickest way to see why a
region of a mesh came out badly -- holes usually trace back to a confidence
map, and smeared edges to depth the filters did or did not reject.

    python scripts/browse_frames.py                       # a RawRecorder session
    python scripts/browse_frames.py --arkitscenes --video-id 42444946

Left/Right step a frame, Shift speeds that up, Home/End jump to the ends, and
the slider scrubs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data import Sequence
from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth

DEFAULT_SESSION = r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
DEFAULT_ARKITSCENES = r"C:\Users\user\work\ARKit3d\sample_data\Validation"

# ARKit grades every depth sample low / medium / high; the fixed colour scale
# keeps those levels comparable from frame to frame.
CONFIDENCE_COLOURS = ["#b2182b", "#f6c141", "#1a9850"]


class FrameBrowser:
    def __init__(self, sequence, max_depth: float, min_confidence: int):
        self.sequence = sequence
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.index = 0

        self.figure, self.axes = plt.subplots(2, 2, figsize=(13, 9))
        self.figure.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.12, hspace=0.18)
        self.axes = self.axes.ravel()
        for axis in self.axes:
            axis.set_xticks([])
            axis.set_yticks([])

        slider_axis = self.figure.add_axes([0.12, 0.04, 0.76, 0.03])
        self.slider = Slider(
            slider_axis, "frame", 0, max(len(sequence) - 1, 1), valinit=0, valstep=1
        )
        self.slider.on_changed(self._on_slider)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)

        self.images: list = [None] * 4
        self.colourbars_done = False
        self.draw()

    # -- interaction ---------------------------------------------------------

    def _on_slider(self, value) -> None:
        index = int(value)
        if index != self.index:
            self.index = index
            self.draw()

    def _on_key(self, event) -> None:
        step = 10 if (event.key or "").startswith("shift+") else 1
        key = (event.key or "").replace("shift+", "")

        if key == "right":
            target = min(self.index + step, len(self.sequence) - 1)
        elif key == "left":
            target = max(self.index - step, 0)
        elif key == "home":
            target = 0
        elif key == "end":
            target = len(self.sequence) - 1
        else:
            return

        if target != self.index:
            # Writing through the slider keeps it in step with the keyboard.
            self.slider.set_val(target)

    # -- drawing -------------------------------------------------------------

    def draw(self) -> None:
        frame = self.sequence[self.index]
        rgb = self.sequence.load_rgb(frame)
        depth = self.sequence.load_depth(frame)
        confidence = self.sequence.load_confidence(frame)
        cleaned, report = clean_depth(
            depth, confidence, min_confidence=self.min_confidence, max_depth=self.max_depth
        )

        valid = depth > 0
        shown = np.where(valid, depth, np.nan)
        shown_clean = np.where(cleaned > 0, cleaned, np.nan)

        for axis in self.axes:
            axis.clear()
            axis.set_xticks([])
            axis.set_yticks([])

        self.axes[0].imshow(rgb)
        self.axes[0].set_title(f"RGB  {rgb.shape[1]}x{rgb.shape[0]}", fontsize=10)

        first = self.axes[1].imshow(shown, cmap="turbo", vmin=0, vmax=self.max_depth)
        self.axes[1].set_title(
            f"depth  {depth.shape[1]}x{depth.shape[0]}   "
            f"{np.nanmin(shown):.2f}-{np.nanmax(shown):.2f} m",
            fontsize=10,
        )

        if confidence is not None:
            self.axes[2].imshow(
                confidence,
                cmap=matplotlib.colors.ListedColormap(CONFIDENCE_COLOURS),
                vmin=0,
                vmax=2,
            )
            counts = np.bincount(confidence.ravel(), minlength=3)
            share = counts / counts.sum()
            self.axes[2].set_title(
                f"confidence   low {share[0]:.0%}  med {share[1]:.0%}  high {share[2]:.0%}",
                fontsize=10,
            )
        else:
            self.axes[2].text(
                0.5, 0.5, "no confidence maps\nin this capture",
                ha="center", va="center", fontsize=11, color="#888",
            )
            self.axes[2].set_title("confidence", fontsize=10)

        self.axes[3].imshow(shown_clean, cmap="turbo", vmin=0, vmax=self.max_depth)
        rejected = {k: v for k, v in report.items() if k not in ("kept", "kept_fraction")}
        summary = "  ".join(f"{k.split('_')[0]} {int(v):,}" for k, v in rejected.items() if v)
        self.axes[3].set_title(
            f"cleaned  kept {report['kept_fraction']:.1%}" + (f"   rejected: {summary}" if summary else ""),
            fontsize=10,
        )

        if not self.colourbars_done:
            self.figure.colorbar(
                first, ax=[self.axes[1], self.axes[3]], shrink=0.7, label="metres"
            )
            self.colourbars_done = True

        name = getattr(frame, "folder", None)
        label = name.name if name is not None else getattr(frame, "key", str(self.index))
        self.figure.suptitle(
            f"frame {self.index + 1} / {len(self.sequence)}   {label}", fontsize=12
        )
        self.figure.canvas.draw_idle()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--arkitscenes", action="store_true", help="read an ARKitScenes capture instead")
    parser.add_argument("--root", default=DEFAULT_ARKITSCENES)
    parser.add_argument("--video-id", default="42444946")
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--min-confidence", type=int, default=1)
    args = parser.parse_args()

    if args.arkitscenes:
        sequence = Sequence(args.root, args.video_id)
        print(f"ARKitScenes {args.video_id}: {len(sequence)} frames")
    else:
        sequence = RawRecordingSequence(args.session)
        print(f"{Path(args.session).name}: {len(sequence)} frames")

    print("left/right = step, shift+arrow = x10, home/end = jump, slider = scrub")
    FrameBrowser(sequence, args.max_depth, args.min_confidence)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
