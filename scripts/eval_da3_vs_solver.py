"""Head-to-head: does a monocular network beat guided upsampling here?

Both methods produce dense depth from the same frame, but they start from very
different places. The bilateral solver interpolates between LiDAR samples and
never sees the scene otherwise. Depth Anything predicts geometry from the image
with no depth input at all, then gets anchored to whatever LiDAR is available.

Whether the network is worth its cost depends entirely on how dense the LiDAR
already is. At 256x192 the samples may simply be close enough together that
interpolating between them beats predicting from scratch -- in which case the
network only earns its place in the gaps, where there is nothing to interpolate.

The test hides a fraction of the LiDAR from both methods, lets each produce a
dense field from what remains, and scores them at the hidden pixels. Results are
split by how far each hidden pixel sits from the nearest surviving sample, since
that is exactly the axis the two methods should differ along.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.anchoring import anchor_to_measurements
from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample_bilateral_solver


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941")
    parser.add_argument("--da3-dir", default="output/da3")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument(
        "--block",
        type=int,
        default=12,
        help="hide contiguous blocks of this many native pixels rather than "
        "scattered ones. Random holdout leaves a surviving sample within a pixel "
        "or two of everything, so no real gap ever forms and interpolation wins "
        "trivially; genuine LiDAR holes are contiguous",
    )
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--dilate", type=int, default=6,
                        help="radius defining a 'gap': hidden pixels further than this "
                             "from any surviving sample")
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)
    by_name = {sequence[i].folder.name: sequence[i] for i in range(len(sequence))}
    rng = np.random.default_rng(0)

    errors: dict[str, dict[str, list]] = {
        m: {"near": [], "far": []} for m in ("solver", "da3")
    }

    for path in sorted(Path(args.da3_dir).glob("*.npz")):
        frame = by_name.get(path.stem)
        if frame is None:
            continue

        predicted = np.load(path)["predicted"].astype(np.float32)
        height, width = predicted.shape

        rgb = sequence.load_rgb(frame)
        confidence = sequence.load_confidence(frame)
        truth, _ = clean_depth(
            sequence.load_depth(frame), confidence, min_confidence=args.min_confidence
        )

        valid = truth > 0
        rows, cols = np.nonzero(valid)
        if len(rows) < 500:
            continue

        # Block mask on a coarse grid, then expanded: this carves out holes with
        # interiors, which is what a confidence-masked region actually looks like.
        block = max(1, args.block)
        gh = int(np.ceil(truth.shape[0] / block))
        gw = int(np.ceil(truth.shape[1] / block))
        coarse = rng.random((gh, gw)) < args.holdout
        hidden_mask = np.kron(coarse, np.ones((block, block), bool))[
            : truth.shape[0], : truth.shape[1]
        ]
        hide = hidden_mask[rows, cols]
        if hide.sum() < 100 or (~hide).sum() < 100:
            continue

        partial = truth.copy()
        partial[rows[hide], cols[hide]] = 0.0
        partial_conf = confidence.copy() if confidence is not None else None
        if partial_conf is not None:
            partial_conf[rows[hide], cols[hide]] = 0

        # Both methods see exactly the same reduced LiDAR.
        solver = upsample_bilateral_solver(partial, rgb, partial_conf, width, height)
        solver, _ = anchor_to_measurements(
            solver,
            cv2.resize(partial, (width, height), interpolation=cv2.INTER_NEAREST),
            cv2.resize(partial_conf, (width, height), interpolation=cv2.INTER_NEAREST)
            if partial_conf is not None else None,
            args.min_confidence,
            fit_scale=False,  # already metric: it interpolates measurements
        )
        da3, _ = anchor_to_measurements(
            predicted,
            cv2.resize(partial, (width, height), interpolation=cv2.INTER_NEAREST),
            cv2.resize(partial_conf, (width, height), interpolation=cv2.INTER_NEAREST)
            if partial_conf is not None else None,
            args.min_confidence,
            fit_scale=True,
        )

        # Distance from each pixel to the nearest surviving sample, on the fine
        # grid, so "near" and "far" mean the same thing for both methods.
        support = cv2.resize(
            (partial > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        distance = cv2.distanceTransform((support == 0).astype(np.uint8), cv2.DIST_L2, 3)

        scale_y, scale_x = height / truth.shape[0], width / truth.shape[1]
        hr = np.clip(((rows[hide] + 0.5) * scale_y - 0.5).round().astype(int), 0, height - 1)
        hc = np.clip(((cols[hide] + 0.5) * scale_x - 0.5).round().astype(int), 0, width - 1)
        actual = truth[rows[hide], cols[hide]]
        far = distance[hr, hc] > args.dilate

        for name, field in (("solver", solver), ("da3", da3)):
            got = field[hr, hc] > 0
            err = np.abs(field[hr, hc] - actual)
            errors[name]["near"].append(err[got & ~far])
            errors[name]["far"].append(err[got & far])

    header = f"{'method':>8} | {'near a sample':>16} | {'in a gap':>14} | {'gap pixels':>11}"
    print(header)
    print("-" * len(header))
    for name in ("solver", "da3"):
        near = np.concatenate(errors[name]["near"])
        far = np.concatenate(errors[name]["far"])
        share = len(far) / max(len(near) + len(far), 1)
        far_text = f"{np.median(far) * 1000:11.1f} mm" if len(far) else "          --"
        print(f"{name:>8} | {np.median(near) * 1000:13.1f} mm | {far_text} | {share:10.1%}")

    print("\nnear a sample = hidden pixel within a few pixels of surviving LiDAR")
    print(f"in a gap      = further than {args.dilate} px from any surviving sample")
    print("\nInterpolation should win near samples. If the network does not win in the")
    print("gaps either, it has nothing to add at this LiDAR density.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
