"""
compare.py - Compare original vs LiZIP-reconstructed point cloud.

Usage:
    python src/utils/compare.py <original> <reconstructed>

The reconstructed file is produced by running:
    python main.py encode <original> out.lizip --mode cpp
    python main.py decode out.lizip reconstructed.bin --mode cpp
"""

import argparse
import os
import sys

import numpy as np
from colorama import Fore, Style, init
from scipy.spatial import cKDTree

init()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def _load_point_cloud(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "data_loader",
        os.path.join(PROJECT_ROOT, "src", "utils", "data_loader.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_point_cloud(path)


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        # Try strides in order: 5 (NuScenes), 3 (LiZIP reconstructed), 4 (KITTI)
        for stride in (5, 3, 4):
            if raw.size % stride == 0:
                return raw.reshape(-1, stride)[:, :3]
        raise ValueError(f"Cannot infer point stride for {path} (float count={raw.size})")
    return _load_point_cloud(path)[:, :3]


def stats(label, values, unit="mm"):
    print(f"  {Fore.CYAN}{label:<20}{Style.RESET_ALL}"
          f"  mean={Fore.YELLOW}{values.mean():.4f}{Style.RESET_ALL}{unit}"
          f"  max={Fore.RED}{values.max():.4f}{Style.RESET_ALL}{unit}"
          f"  p95={values[int(len(values)*0.95)]:.4f}{unit}"
          f"  p99={values[int(len(values)*0.99)]:.4f}{unit}")


def compare(original_path, reconstructed_path):
    print(f"\n{Fore.RED}{Style.BRIGHT}LiZIP Reconstruction Error Report{Style.RESET_ALL}")
    print(f"  Original      : {original_path}")
    print(f"  Reconstructed : {reconstructed_path}\n")

    orig = load(original_path)
    rec  = load(reconstructed_path)

    print(f"  {Fore.CYAN}Original points   :{Style.RESET_ALL} {len(orig):,}")
    print(f"  {Fore.CYAN}Reconstructed pts :{Style.RESET_ALL} {len(rec):,}")

    orig_size = os.path.getsize(original_path)
    rec_size  = os.path.getsize(reconstructed_path)
    print(f"  {Fore.CYAN}Original size     :{Style.RESET_ALL} {orig_size:,} bytes")
    print(f"  {Fore.CYAN}Reconstructed size:{Style.RESET_ALL} {rec_size:,} bytes\n")

    # nearest-neighbour distances: reconstructed -> original
    tree = cKDTree(orig[:, :3])
    dists_rec_to_orig, _ = tree.query(rec[:, :3], k=1)
    dists_rec_to_orig *= 1000.0  # convert to mm

    # nearest-neighbour distances: original -> reconstructed (symmetric check)
    tree2 = cKDTree(rec[:, :3])
    dists_orig_to_rec, _ = tree2.query(orig[:, :3], k=1)
    dists_orig_to_rec *= 1000.0

    dists_rec_to_orig.sort()
    dists_orig_to_rec.sort()

    print(f"{Fore.RED}{Style.BRIGHT}  Reconstructed -> Original{Style.RESET_ALL}")
    stats("nearest-neighbour", dists_rec_to_orig)

    print(f"\n{Fore.RED}{Style.BRIGHT}  Original -> Reconstructed{Style.RESET_ALL}")
    stats("nearest-neighbour", dists_orig_to_rec)

    chamfer = (dists_rec_to_orig.mean() + dists_orig_to_rec.mean()) / 2.0
    hausdorff = max(dists_rec_to_orig.max(), dists_orig_to_rec.max())

    print(f"\n{Fore.RED}{Style.BRIGHT}  Summary{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{'Chamfer distance':<20}{Style.RESET_ALL}  {Fore.YELLOW}{chamfer:.4f}{Style.RESET_ALL} mm")
    print(f"  {Fore.CYAN}{'Hausdorff distance':<20}{Style.RESET_ALL}  {Fore.RED}{hausdorff:.4f}{Style.RESET_ALL} mm")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="compare",
        description="Compare original vs LiZIP-reconstructed point cloud"
    )
    parser.add_argument("original", help="Original point cloud (.bin, .txt, .ply)")
    parser.add_argument("reconstructed", help="Reconstructed point cloud (.bin, .txt, .ply)")
    args = parser.parse_args()

    for p in (args.original, args.reconstructed):
        if not os.path.isfile(p):
            print(f"{Fore.RED}[error]{Style.RESET_ALL} File not found: {p}")
            sys.exit(1)

    compare(args.original, args.reconstructed)


if __name__ == "__main__":
    main()
