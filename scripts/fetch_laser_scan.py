"""Download one ARKitScenes Faro laser scan, the dataset's true ground truth.

Every other number in this repo compares a reconstruction against the LiDAR that
built it, which measures agreement with the sensor rather than with reality. The
laser scans are independent of the capture entirely, so they are the only thing
here that can measure absolute accuracy.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"


def download(url: str, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  already have {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")
        return True

    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with partial.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {destination.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB"
                              f"  ({done / total:5.1%})", end="", flush=True)
        print()
    except Exception as error:
        print(f"\n  FAILED {destination.name}: {error}")
        partial.unlink(missing_ok=True)
        return False

    # Rename only once complete, so an interrupted run cannot leave a truncated
    # file that later looks valid.
    partial.replace(destination)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visit-id", default="421337")
    parser.add_argument("--scan-id", default="170078")
    parser.add_argument("--out", default=r"C:\Users\user\data\arkitscenes_gt")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.visit_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"visit {args.visit_id}, scan {args.scan_id} -> {out_dir}")

    ok = True
    for suffix in (".ply", "_pose.txt"):
        name = f"{args.scan_id}{suffix}"
        url = f"{BASE}/raw/laser_scanner_point_clouds/{args.visit_id}/{name}"
        # The pose file registers multiple scans into a common frame; it is
        # absent for single-position visits, so its failure is not fatal.
        if not download(url, out_dir / name) and suffix == ".ply":
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
