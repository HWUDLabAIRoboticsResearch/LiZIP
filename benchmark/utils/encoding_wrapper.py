
import os
import sys
import time
import struct
import zlib
import lzma
import gzip
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(SCRIPT_DIR)

from suppress import suppress_stdout
from model import PointPredictorMLP
from data_loader import load_point_cloud
import encoder

import argparse

try:
    import DracoPy
except ImportError:
    DracoPy = None
    print("Warning: DracoPy not found.")

try:
    import laspy
except ImportError:
    laspy = None
    print("Warning: laspy not found.")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "benchmark_out")

def get_lidar_dir(dataset_name):
    if dataset_name.lower() == 'kitti':
        return os.path.join(PROJECT_ROOT, "data", "KITTI")
    elif dataset_name.lower() == 'argoverse':
        return os.path.join(PROJECT_ROOT, "data", "argoverse")
    else:
        return os.path.join(PROJECT_ROOT, "data", "nuScenes", "LIDAR_TOP")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_extended_v2.pth")
CONTEXT_SIZE = 5
HIDDEN_DIM = 512
RESID_SCALE = 100000.0
BLOCK_SIZE = 128
WARMUP_FRAMES = 5
TOTAL_FRAMES = 55

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_files(lidar_dir, count=55, randomize=False):
    import glob
    import random
    files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))
    if not files:
        files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))
    if not files:
        files = sorted(glob.glob(os.path.join(lidar_dir, "*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    
    if randomize:
        if len(files) > count:
            return random.sample(files, count)
        return random.sample(files, len(files))
    
    return files[:count]

def encode_lizip(model, input_path, output_path, compression='zlib', gt_out_path=None):
    start_t = time.time()
    with suppress_stdout():
        _, _ = encoder.encode_file_closed_loop(input_path, output_path, model, compression=compression, debug_gt_path=gt_out_path)
    
    enc_time = time.time() - start_t
    return enc_time

def encode_draco(input_path, output_path, quantization_bits=24):
    if DracoPy is None: return 0
    points = load_point_cloud(input_path)[:, :3]
    
    start_t = time.time()
    drc_data = DracoPy.encode(points.flatten(), quantization_bits=quantization_bits, compression_level=1, create_metadata=False, preserve_order=True)
    with open(output_path, 'wb') as f:
        f.write(drc_data)
    return time.time() - start_t

def encode_laszip(input_path, output_path, scales=[0.00001, 0.00001, 0.00001]):
    if laspy is None: return 0
    points = load_point_cloud(input_path)
    
    start_t = time.time()
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = scales
    header.offsets = [0, 0, 0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:,0], points[:,1], points[:,2]
    las.write(output_path)
    return time.time() - start_t

def encode_gzip(input_path, output_path):
    points = load_point_cloud(input_path)
    
    start_t = time.time()
    with gzip.open(output_path, 'wb') as f:
        f.write(points.tobytes())
    return time.time() - start_t

def benchmark_encoding(model, files, output_dir):
    ensure_dir(output_dir)
    
    results = {
        "LiZIP (Python, zlib)": [],
        "LiZIP (Python, lzma)": [],
        "Draco (Lossy)": [],
        "Draco (Lossless)": [],
        "LASzip (Lossy)": [],
        "LASzip (Lossless)": [],
        "GZip": []
    }
    
    print(f"Starting Encoding Benchmark on {len(files)} frames...")
    
    for i, fpath in tqdm(enumerate(files), total=len(files), desc="Encoding", unit="frame"):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        # print(f"Encoding {i+1}/{len(files)}: {fname}")
        
        p_lizip_zlib = os.path.join(output_dir, f"{fname}.zlib.lizip")
        p_lizip_lzma = os.path.join(output_dir, f"{fname}.lzma.lizip")
        p_draco_lossy = os.path.join(output_dir, f"{fname}.lossy.drc")
        p_draco_lossless = os.path.join(output_dir, f"{fname}.lossless.drc")
        p_las_lossy   = os.path.join(output_dir, f"{fname}.lossy.laz")
        p_las_lossless   = os.path.join(output_dir, f"{fname}.lossless.laz")
        p_gzip  = os.path.join(output_dir, f"{fname}.gz")
        
        # LiZIP (zlib)
        t = encode_lizip(model, fpath, p_lizip_zlib, compression='zlib')
        results["LiZIP (Python, zlib)"].append(t)

        # LiZIP (lzma)
        t = encode_lizip(model, fpath, p_lizip_lzma, compression='lzma')
        results["LiZIP (Python, lzma)"].append(t)
        
        # Draco (Lossy: 14 bits)
        t = encode_draco(fpath, p_draco_lossy, quantization_bits=14)
        results["Draco (Lossy)"].append(t)
        
        # Draco (Lossless: 24 bits)
        t = encode_draco(fpath, p_draco_lossless, quantization_bits=24)
        results["Draco (Lossless)"].append(t)
        
        # Laszip (Lossy: 0.001)
        t = encode_laszip(fpath, p_las_lossy, scales=[0.001, 0.001, 0.001])
        results["LASzip (Lossy)"].append(t)

        # Laszip (Lossless: 0.000001)
        t = encode_laszip(fpath, p_las_lossless, scales=[1e-6, 1e-6, 1e-6])
        results["LASzip (Lossless)"].append(t)
        
        # GZip
        t = encode_gzip(fpath, p_gzip)
        results["GZip"].append(t)
        
    return results

def plot_encoding_results(results, warmup=0, dataset_name="NuScenes"):
    sliced_results = {}
    for k, v in results.items():
        if "Python" in k: continue
        
        if len(v) > warmup:
            sliced_results[k] = v[warmup:]
        else:
            sliced_results[k] = []
            
    if not any(sliced_results.values()):
        print("Not enough data to plot.")
        return

    ds_display = dataset_name.capitalize() if dataset_name else "NuScenes"
    plt.figure(figsize=(12, 6))
    
    colors = {
        'LiZIP (C++, zlib)': 'darkred',
        'LiZIP (C++, lzma)': 'blue',
        'Draco': 'skyblue', 
        'Laszip': 'green',
        'GZip': 'gray'
    }
    markers = {
        'LiZIP (C++, zlib)': 's',
        'LiZIP (C++, lzma)': 'D',
        'Draco': 'x', 
        'Laszip': 'p',
        'GZip': '^'
    }
    
    first_key = next(iter(sliced_results))
    x_axis = np.arange(warmup + 1, len(sliced_results[first_key]) + warmup + 1)
    
    for method, times in sliced_results.items():
        if not times: continue
        display_label = f"{method} (Ours)" if "LiZIP" in method else method
        
        color = 'black'
        for key in colors:
            if key in method:
                color = colors[key]
                break
        
        marker = '.'
        for key in markers:
            if key in method:
                marker = markers[key]
                break

        plt.plot(x_axis, times, label=display_label, color=color, marker=marker)

    plt.title(f"Encoding Time per Frame ({ds_display})")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_file = os.path.join(SCRIPT_DIR, "graphs", "encoding_benchmark.png")
    ensure_dir(os.path.dirname(out_file))
    plt.savefig(out_file)
    print(f"Encoding graph saved to {out_file}")

MODEL_BIN = os.path.join(PROJECT_ROOT, "models", "grid_search", "mlp_c3_h256.bin")
WARMUP_FRAMES = 0
TOTAL_FRAMES = 100

def main():
    parser = argparse.ArgumentParser(description="Run Encoding Benchmark")
    parser.add_argument('--dataset', type=str, default='nuscenes', choices=['nuscenes', 'kitti', 'argoverse'], help='Dataset to use (nuscenes, kitti, argoverse)')
    args = parser.parse_args()

    lidar_dir = get_lidar_dir(args.dataset)
    print(f"Using LiDAR directory: {lidar_dir}")

    encoder.device = device
    
    print(f"Loading model: {MODEL_PATH}")
    model = PointPredictorMLP(context_size=CONTEXT_SIZE, hidden_dim=HIDDEN_DIM)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    files = get_files(lidar_dir, TOTAL_FRAMES)
    if len(files) < TOTAL_FRAMES:
        print(f"Warning: Only found {len(files)} files, expected {TOTAL_FRAMES}.")
    
    results = benchmark_encoding(model, files, OUTPUT_DIR)
    plot_encoding_results(results, WARMUP_FRAMES, dataset_name=args.dataset)
    
    print("\nEncoding Stats (Mean +/- Std):")
    for k, v in results.items():
        if len(v) > WARMUP_FRAMES:
            valid = v[WARMUP_FRAMES:]
            print(f"{k:<10}: {np.mean(valid):.4f} +/- {np.std(valid):.4f} s")

if __name__ == "__main__":
    main()
