"""Precompute Depth Anything 3 monocular predictions for an ICL-NUIM sequence.

Run once under the `depth` conda env (torch cu128) -- `indoor3d`'s own env
never needs torch, since `indoor3d.fusion.fuse` reads these predictions back
through `indoor3d.da3_cache.CachedDepthProvider`. See that module for the
on-disk format. Resumable: existing cache files are skipped unless
`--overwrite` is given, so a killed or interrupted run just continues.

    conda activate depth
    python scripts/precompute_da3_depth.py --frames all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from indoor3d.da3_cache import cache_path, write_depth
from indoor3d.data_iclnuim import ICLNUIMSequence


def _parse_frames(spec: str, available: list[int]) -> list[int]:
    """Parse spec as 'all', 'lo-hi', or comma-separated ids; filter to what exists."""
    if spec == "all":
        return available
    if "-" in spec:
        lo_str, hi_str = spec.split("-", 1)
        lo, hi = int(lo_str), int(hi_str)
        return [frame_id for frame_id in available if lo <= frame_id <= hi]
    wanted = {int(token) for token in spec.split(",")}
    return [frame_id for frame_id in available if frame_id in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=r"C:\Users\user\data\iclnuim")
    parser.add_argument(
        "--cache-dir", default=None, help="default: <root>/da3_depth_cache"
    )
    parser.add_argument(
        "--frames",
        default="all",
        help='"all", "lo-hi" (inclusive), or a comma list, e.g. "400,401,402"',
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute frames already cached"
    )
    parser.add_argument(
        "--progress-every", type=int, default=20, help="0 disables progress printing"
    )
    args = parser.parse_args()

    # Deferred: only needed once we know there is work to do, and this import
    # is what requires the `depth` conda env rather than `indoor3d`'s.
    from depth_anything_3.api import DepthAnything3

    sequence = ICLNUIMSequence(args.root)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else Path(args.root) / "da3_depth_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    available = [frame.frame_id for frame in sequence.frames]
    frame_ids = _parse_frames(args.frames, available)
    print(f"{len(frame_ids)} of {len(available)} frames selected")

    todo = [
        frame_id
        for frame_id in frame_ids
        if args.overwrite or not cache_path(cache_dir, frame_id).is_file()
    ]
    print(f"{len(todo)} not yet cached at {cache_dir}")
    if not todo:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading DA3METRIC-LARGE onto {device}...", flush=True)
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(
        device=device
    )

    by_id = {frame.frame_id: frame for frame in sequence.frames}
    for done, frame_id in enumerate(todo, start=1):
        frame = by_id[frame_id]
        rgb = sequence.load_rgb(frame)
        _, width, height = sequence.intrinsics(frame)

        prediction = model.inference([rgb], process_res=args.process_res)
        predicted = np.asarray(prediction.depth)[0].astype(np.float32)
        predicted_full = cv2.resize(
            predicted, (width, height), interpolation=cv2.INTER_CUBIC
        )
        write_depth(cache_path(cache_dir, frame_id), predicted_full)

        if args.progress_every and (
            done % args.progress_every == 0 or done == len(todo)
        ):
            print(f"  {done}/{len(todo)} cached (frame {frame_id})", flush=True)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
