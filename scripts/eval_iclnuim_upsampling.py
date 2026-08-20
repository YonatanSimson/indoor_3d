"""Does guided upsampling recover real structure, or just interpolate noise?

Stage B question 2 from `hand_off_original_plan.md`. `eval_upsampling_accuracy.py`
already holds out samples and checks whether upsampling predicts them -- but
"them" is samples of the *same noisy field* the method was fed, so a method
that is merely self-consistent with its own input passes that test. This is
the stronger version the hand-off doc asks for: ICL-NUIM ships the true
640x480 depth the simulated 256x192 input was downsampled *from*, so the
comparison is against real geometry, not against more of the same noisy
samples.

For each frame: degrade the true depth to our sensor's 256x192 (same
`sensor_sim.degrade_frame` the rest of Phase V uses), run it through
`clean_depth` exactly as `fusion.fuse` does before upsampling ever sees it --
so low-confidence and flying-pixel rejections leave real gaps here too, not
just the downsampling-support holes `degrade_frame` produces on its own --
then upsample back to 640x480 with each method and compare against the
*true* clean depth at every pixel, not just the ones the method was shown.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_iclnuim import ICLNUIMSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample
from indoor3d.sensor_sim import SensorSimConfig, degrade_frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=r"C:\Users\user\data\iclnuim")
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument(
        "--methods", nargs="+", default=["bilinear", "joint", "solver", "anchored"]
    )
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sequence = ICLNUIMSequence(args.root)
    indices = np.linspace(0, len(sequence) - 1, args.num_frames, dtype=int)
    rng = np.random.default_rng(args.seed)
    sensor_config = SensorSimConfig()

    print(f"{len(indices)} frames, true 640x480 vs simulated-256x192-then-upsampled\n")

    errors: dict[str, list[np.ndarray]] = {m: [] for m in args.methods}
    filled: dict[str, list[bool]] = {m: [] for m in args.methods}

    for index in indices:
        frame = sequence[index]
        true_depth = sequence.load_depth(frame)  # native 640x480, clean
        rgb = sequence.load_rgb(frame)
        intrinsics, width, height = sequence.intrinsics(frame)

        degraded_depth, confidence, _ = degrade_frame(
            true_depth, rgb, intrinsics, sensor_config, rng
        )
        # Same cleanup fuse() applies before upsampling: confidence gating
        # and flying-pixel rejection leave real holes, not just the
        # downsampling-support gaps degrade_frame produces on its own.
        cleaned_depth, _ = clean_depth(
            degraded_depth, confidence, min_confidence=args.min_confidence
        )

        for method in args.methods:
            upsampled = upsample(cleaned_depth, rgb, confidence, width, height, method)

            valid_true = true_depth > 0
            got = (upsampled > 0) & valid_true
            filled[method].append(float(got.sum()) / max(1, valid_true.sum()))
            if got.any():
                errors[method].append(np.abs(upsampled[got] - true_depth[got]))

    header = (
        f"{'method':>10} | {'median':>10} | {'mean':>8} | {'p90':>8} | {'coverage':>9}"
    )
    print(header)
    print("-" * len(header))
    for method in args.methods:
        if not errors[method]:
            print(f"{method:>10} | no data")
            continue
        all_errors = np.concatenate(errors[method])
        coverage = float(np.mean(filled[method]))
        print(
            f"{method:>10} | {np.median(all_errors) * 1000:8.1f}mm | "
            f"{all_errors.mean() * 1000:6.1f}mm | {np.percentile(all_errors, 90) * 1000:6.1f}mm | "
            f"{coverage:8.1%}"
        )

    print(
        "\nmedian/mean/p90 = |upsampled - true| at every true-depth pixel (mm, smaller is better)"
    )
    print(
        "coverage        = fraction of true-depth pixels the method actually predicted"
    )
    print(
        "\nThis is the genuine held-out test: true 640x480 depth was never shown to any"
    )
    print("method above, only the degraded 256x192 version of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
