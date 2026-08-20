"""Does guided upsampling get the depth *numbers* right, not just the edges?

Edge-alignment metrics reward putting a discontinuity in the right place. They
say nothing about whether the depth values are correct, and a global solver has
every opportunity to move them: it smooths, it propagates, and it extrapolates
into holes. Sharper and wronger is a real possibility.

So this holds measurements out. A fraction of the real depth samples are hidden,
each method upsamples from what remains, and the prediction at the hidden pixels
is compared against the values that were actually measured there. Ground truth
is the sensor itself, which is the right reference for this question: we are
asking whether upsampling reproduces the measurements it was not shown.

Two numbers per method:

- held-out error : accuracy at pixels the method never saw. The real test.
- observed drift : how far it moved depths it *was* given. Upsampling is not
                   supposed to overwrite measurements; large drift means the
                   smoothing prior is overpowering the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample


def sample_at(
    image: np.ndarray, rows: np.ndarray, cols: np.ndarray, scale: float
) -> np.ndarray:
    """Read an upsampled image at the centres of the original pixels."""
    r = np.clip(((rows + 0.5) * scale - 0.5).round().astype(int), 0, image.shape[0] - 1)
    c = np.clip(((cols + 0.5) * scale - 0.5).round().astype(int), 0, image.shape[1] - 1)
    return image[r, c]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
    )
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument(
        "--holdout", type=float, default=0.25, help="fraction of samples hidden"
    )
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["bilinear", "joint", "solver"])
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)
    indices = np.linspace(0, len(sequence) - 1, args.num_frames, dtype=int)
    rng = np.random.default_rng(0)

    held: dict[str, list[np.ndarray]] = {m: [] for m in args.methods}
    drift: dict[str, list[np.ndarray]] = {m: [] for m in args.methods}
    missing: dict[str, int] = {m: 0 for m in args.methods}
    total_held = 0

    print(
        f"{len(indices)} frames, x{args.scale:g}, hiding {args.holdout:.0%} of samples\n"
    )

    for index in indices:
        frame = sequence[index]
        rgb = sequence.load_rgb(frame)
        confidence = sequence.load_confidence(frame)
        truth, _ = clean_depth(
            sequence.load_depth(frame), confidence, min_confidence=args.min_confidence
        )

        valid = truth > 0
        rows, cols = np.nonzero(valid)
        if len(rows) < 500:
            continue

        hide = rng.random(len(rows)) < args.holdout
        hidden_rows, hidden_cols = rows[hide], cols[hide]
        kept_rows, kept_cols = rows[~hide], cols[~hide]

        # The method sees everything except the hidden pixels.
        partial = truth.copy()
        partial[hidden_rows, hidden_cols] = 0.0

        partial_confidence = None
        if confidence is not None:
            partial_confidence = confidence.copy()
            partial_confidence[hidden_rows, hidden_cols] = 0

        width = int(truth.shape[1] * args.scale)
        height = int(truth.shape[0] * args.scale)
        total_held += len(hidden_rows)

        for method in args.methods:
            out = upsample(partial, rgb, partial_confidence, width, height, method)

            predicted = sample_at(out, hidden_rows, hidden_cols, args.scale)
            actual = truth[hidden_rows, hidden_cols]
            # A method that leaves a hole empty has not predicted anything;
            # count that separately rather than scoring it as a huge error.
            got = predicted > 0
            missing[method] += int((~got).sum())
            if got.any():
                held[method].append(np.abs(predicted[got] - actual[got]))

            observed = sample_at(out, kept_rows, kept_cols, args.scale)
            kept_truth = truth[kept_rows, kept_cols]
            ok = observed > 0
            if ok.any():
                drift[method].append(np.abs(observed[ok] - kept_truth[ok]))

    header = (
        f"{'method':>10} | {'held-out median':>16} | {'held-out mean':>14} | "
        f"{'no prediction':>14} | {'observed drift':>15}"
    )
    print(header)
    print("-" * len(header))
    for method in args.methods:
        if not held[method]:
            print(f"{method:>10} | no data")
            continue
        h = np.concatenate(held[method])
        d = np.concatenate(drift[method]) if drift[method] else np.array([np.nan])
        print(
            f"{method:>10} | {np.median(h)*1000:13.1f} mm | {h.mean()*1000:11.1f} mm | "
            f"{missing[method]/max(total_held,1):13.1%} | {np.median(d)*1000:12.1f} mm"
        )

    print(
        f"\nheld-out    = error at {args.holdout:.0%} of samples the method never saw"
    )
    print(
        "observed drift = how far it moved depths it WAS given (lower = respects the data)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
