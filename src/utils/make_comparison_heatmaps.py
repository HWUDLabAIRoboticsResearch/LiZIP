"""
make_comparison_heatmaps.py - Rotating reconstruction error heatmaps for
LiZIP, LASzip and Draco, all rendered on a shared colormap scale.

Usage:
    python src/utils/make_comparison_heatmaps.py <input.bin> [--frames 60] [--fps 20]

Output (benchmark/gifs/):
    error_heatmap_lizip.gif
    error_heatmap_laszip.gif
    error_heatmap_draco.gif
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile

import DracoPy
import imageio
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "gifs")


def load_bin(path):
    raw = np.fromfile(path, dtype=np.float32)
    for stride in (5, 3, 4):
        if raw.size % stride == 0:
            return raw.reshape(-1, stride)[:, :3]
    raise ValueError(f"Cannot infer point stride for {path}")


def roundtrip_lizip(orig, input_path):
    tmp_lizip = tempfile.mktemp(suffix=".lizip")
    tmp_rec   = tempfile.mktemp(suffix=".bin")
    try:
        main_py = os.path.join(PROJECT_ROOT, "main.py")
        subprocess.run(
            [sys.executable, main_py, "encode", input_path, tmp_lizip, "--mode", "cpp"],
            check=True, capture_output=True
        )
        subprocess.run(
            [sys.executable, main_py, "decode", tmp_lizip, tmp_rec, "--mode", "cpp"],
            check=True, capture_output=True
        )
        return load_bin(tmp_rec)
    finally:
        for f in (tmp_lizip, tmp_rec):
            if os.path.exists(f):
                os.remove(f)


def roundtrip_laszip(orig):
    """Write with 0.01 mm (1e-5 m) scale to match LiZIP quantisation precision."""
    tmp = tempfile.mktemp(suffix=".las")
    try:
        header = laspy.LasHeader(point_format=0, version="1.2")
        header.offsets = orig.min(axis=0)
        header.scales  = np.array([1e-5, 1e-5, 1e-5])
        las = laspy.LasData(header=header)
        las.x = orig[:, 0].astype(np.float64)
        las.y = orig[:, 1].astype(np.float64)
        las.z = orig[:, 2].astype(np.float64)
        las.write(tmp)
        las2 = laspy.read(tmp)
        return np.stack([np.array(las2.x), np.array(las2.y), np.array(las2.z)], axis=1)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def roundtrip_draco(orig, quantization_bits=24):
    enc = DracoPy.encode_point_cloud_to_buffer(
        orig.astype(np.float32), quantization_bits=quantization_bits
    )
    dec = DracoPy.decode(enc)
    return dec._attributes[0]["data"].reshape(-1, 3)


def nn_error_mm(orig, rec):
    tree  = cKDTree(rec)
    dists, _ = tree.query(orig, k=1)
    return dists * 1000.0


def render_frame(orig, errors, elev, azim, vmin, vmax, title):
    fig = plt.figure(figsize=(6, 5), dpi=110)
    fig.patch.set_facecolor("#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")

    sc = ax.scatter(
        orig[:, 0], orig[:, 1], orig[:, 2],
        c=errors, cmap="plasma", vmin=vmin, vmax=vmax,
        s=0.35, linewidths=0, alpha=0.85
    )

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    margin = 0.5
    ax.set_xlim(orig[:, 0].min() - margin, orig[:, 0].max() + margin)
    ax.set_ylim(orig[:, 1].min() - margin, orig[:, 1].max() + margin)
    ax.set_zlim(orig[:, 2].min() - margin, orig[:, 2].max() + margin)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.025, shrink=0.6)
    cbar.set_label("NN error (mm)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=11, pad=6)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return imageio.v2.imread(buf)


def make_gif(orig, errors, vmin, vmax, title, out_path, n_frames, elev, fps):
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    frames   = []
    for i, azim in enumerate(azimuths):
        print(f"\r    frame {i+1}/{n_frames} ...", end="", flush=True)
        frames.append(render_frame(orig, errors, elev, azim, vmin, vmax, title))
    print()
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"    saved -> {out_path}  ({os.path.getsize(out_path)//1024} KB)")


def main():
    parser = argparse.ArgumentParser(description="Comparison error heatmap GIFs")
    parser.add_argument("input",   help="Original .bin point cloud")
    parser.add_argument("--frames", type=int,   default=60)
    parser.add_argument("--fps",    type=int,   default=20)
    parser.add_argument("--elev",   type=float, default=22)
    parser.add_argument("--qbits",  type=int,   default=24, help="Draco quantization bits")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[error] File not found: {args.input}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {args.input}")
    orig = load_bin(args.input)
    print(f"  {len(orig):,} points")

    print("LiZIP  : encoding + decoding ...")
    rec_lizip  = roundtrip_lizip(orig, args.input)

    print("LASzip : encoding + decoding ...")
    rec_laszip = roundtrip_laszip(orig)

    print(f"Draco  : encoding + decoding (quantization_bits={args.qbits}) ...")
    rec_draco  = roundtrip_draco(orig, quantization_bits=args.qbits)

    print("Computing NN errors ...")
    err_lizip  = nn_error_mm(orig, rec_lizip)
    err_laszip = nn_error_mm(orig, rec_laszip)
    err_draco  = nn_error_mm(orig, rec_draco)

    for label, err in [("LiZIP", err_lizip), ("LASzip", err_laszip), ("Draco", err_draco)]:
        print(f"  {label:<8}  mean={err.mean():.4f} mm  max={err.max():.4f} mm  p99={np.percentile(err,99):.4f} mm")

    vmin = 0.0
    vmax = np.percentile(err_draco, 99)
    print(f"\nShared colormap scale: 0.0 - {vmax:.4f} mm  (Draco p99)")

    methods = [
        ("LiZIP",  err_lizip,  "Reconstruction Error (LiZIP)",  "error_heatmap_lizip.gif"),
        ("LASzip", err_laszip, "Reconstruction Error (LASzip)", "error_heatmap_laszip.gif"),
        ("Draco",  err_draco,  "Reconstruction Error (Draco)",  "error_heatmap_draco.gif"),
    ]

    for label, err, title, fname in methods:
        print(f"\nRendering {label} ({args.frames} frames) ...")
        out_path = os.path.join(OUT_DIR, fname)
        make_gif(orig, err, vmin, vmax, title, out_path, args.frames, args.elev, args.fps)

    print("\nDone.")


if __name__ == "__main__":
    main()
