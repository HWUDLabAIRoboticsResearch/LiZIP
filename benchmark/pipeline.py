import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import subprocess
import re
from tqdm import tqdm
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

import encoding
import decoding
from model import PointPredictorMLP
from data_loader import load_point_cloud
import encoder 
import decoder
import argparse

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "benchmark_out")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "grid_search", "mlp_c3_h256.pth")
CPP_EXE = os.path.join(PROJECT_ROOT, "lizip.exe")
MODEL_BIN = os.path.join(PROJECT_ROOT, "models", "grid_search", "mlp_c3_h256.bin")
CONTEXT_SIZE = 3
HIDDEN_DIM = 256
TOTAL_FRAMES = 100

def get_lidar_dir(dataset_name):
    if dataset_name.lower() == 'kitti':
        return os.path.join(PROJECT_ROOT, "data", "KITTI")
    elif dataset_name.lower() == 'argoverse':
        return os.path.join(PROJECT_ROOT, "data", "argoverse")
    else:
        return os.path.join(PROJECT_ROOT, "data", "nuScenes", "LIDAR_TOP")

device = torch.device("cpu")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_cpp_lizip(mode, input_path, output_path, model_bin, compression=None):
    temp_bin = None
    try:
        if mode == 'e' and (input_path.endswith('.txt') or input_path.endswith('.ply')):
            points = load_point_cloud(input_path)
            N = points.shape[0]
            padded = np.zeros((N, 5), dtype=np.float32)
            if points.shape[1] >= 4:
                padded[:, :4] = points[:, :4]
            else:
                padded[:, :3] = points[:, :3]
            temp_bin = input_path + ".temp.bin"
            padded.tofile(temp_bin)
            input_arg = temp_bin
        else:
            input_arg = input_path

        args = [CPP_EXE, mode, input_arg, output_path, model_bin]
        if compression:
            args.append(compression)
            
        result = subprocess.run(
            args,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"C++ Error: {result.stderr}")
            return 0.0, {}
            
        match = re.search(r"in ([\d\.]+)s", result.stdout)
        
        breakdown = {}
        for key in ["Raw_Float_Size", "Quantized_Int_Size", "Stage1_Entropy_Only", "Stage2_MLP_Residuals", "Stage3_Final_Shuffled"]:
            b_match = re.search(fr"{key}: (\d+)", result.stdout)
            if b_match:
                breakdown[key] = int(b_match.group(1))

        if temp_bin and os.path.exists(temp_bin):
            os.remove(temp_bin)
            
        return (float(match.group(1)) if match else 0.0), breakdown
    except Exception as e:
        if temp_bin and os.path.exists(temp_bin):
            os.remove(temp_bin)
        return 0.0, {}

def calculate_max_error(gt_points, rec_points):
    if rec_points is None: return 99999.0
    if len(gt_points) == 0: return 0.0
    
    try:
        tree = cKDTree(gt_points)
        dists, _ = tree.query(rec_points, k=1)
        return np.max(dists) * 1000.0
    except Exception as e:
        print(f"Error in KDTree calc: {e}")
        return 99999.0

def plot_pipeline_results(enc_results, dec_results, errors, sizes, dataset_name="NuScenes", breakdown_stats=None):
    START_FRAME = 3
    
    num_frames_total = len(next(iter(enc_results.values())))
    x_axis = np.arange(START_FRAME + 1, num_frames_total + 1)
    
    ds_display = dataset_name.capitalize() if dataset_name else "NuScenes"
    
    methods = [m for m in enc_results.keys() if m != 'GZip']
    
    colors = {
        'LiZIP (Python, zlib)': 'red', 
        'LiZIP (Python, lzma)': 'orange',
        'LiZIP (C++, zlib)': 'darkred',
        'LiZIP (C++, lzma)': 'blue',
        'Draco': 'skyblue', 
        'Laszip': 'green',
        'GZip': 'gray'
    }
    markers = {
        'LiZIP (Python, zlib)': 'o', 
        'LiZIP (Python, lzma)': 'v',
        'LiZIP (C++, zlib)': 's',
        'LiZIP (C++, lzma)': 'D',
        'Draco': 'x', 
        'Laszip': 'p',
        'GZip': '^'
    }
    
    plt.figure(figsize=(12, 6))
    for m in methods:
        enc = np.array(enc_results[m][START_FRAME:])
        dec = np.array(dec_results[m][START_FRAME:])
        total = enc + dec
        display_label = f"{m} (Ours)" if "LiZIP" in m else m
        color = colors.get(m, 'black')
        marker = markers.get(m, '.')
        plt.plot(x_axis, total, label=display_label, color=color, marker=marker)
            
    plt.title(f"Total Pipeline Time per Frame ({ds_display})")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, "graphs", "pipeline_total_time.png"))

    plt.figure(figsize=(12, 6))
    for m in methods:
        sz = np.array(sizes[m][START_FRAME:])
        display_label = f"{m} (Ours)" if "LiZIP" in m else m
        color = colors.get(m, 'black')
        marker = markers.get(m, '.')
        plt.plot(x_axis, sz, label=display_label, color=color, marker=marker)
            
    plt.title(f"Compressed File Size per Frame ({ds_display})")
    plt.xlabel("Frame Index")
    plt.ylabel("Size (KB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, "graphs", "pipeline_file_size.png"))

    plt.figure(figsize=(12, 6))
    for m in methods:
        err = np.array(errors[m][START_FRAME:])
        display_label = f"{m} (Ours)" if "LiZIP" in m else m
        color = colors.get(m, 'black')
        marker = markers.get(m, '.')
        plt.plot(x_axis, err, label=display_label, color=color, marker=marker)
            
    plt.yscale('log')
    plt.title(f"Reconstruction Error (Max L2 Distance) per Frame ({ds_display})")
    plt.xlabel("Frame Index")
    plt.ylabel("Error (mm) [Log Scale]")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, "graphs", "pipeline_error.png"))

    if breakdown_stats:
        plt.figure(figsize=(10, 6))
        stages = ["Raw Float", "Quantized Int", "+ MLP Residuals", "+ Byte Shuffle"]
        keys = ["Raw_Float_Size", "Quantized_Int_Size", "Stage2_MLP_Residuals", "Stage3_Final_Shuffled"]
        avg_sizes = []

        print("\nAverage Size Breakdown per Stage (LiZIP C++ LZMA):")
        for k, stage_name in zip(keys, stages):
            vals = [b[k] for b in breakdown_stats if k in b]
            avg_val = np.mean(vals) if vals else 0
            avg_sizes.append(avg_val)
            print(f"  {stage_name}: {avg_val:.2f} KB")

        plt.bar(stages, avg_sizes, color=['gray', 'silver', 'orange', 'red'])
        plt.ylabel("Average Size (KB)")
        plt.title(f"Size Reduction Waterfall ({ds_display})")
        plt.grid(True, axis='y', alpha=0.3)

        
        for i, val in enumerate(avg_sizes):
            plt.text(i, val + 5, f"{val:.1f} KB", ha='center', fontweight='bold')
            if i > 0:
                reduction = (1 - val/avg_sizes[i-1]) * 100
                plt.text(i, val/2, f"-{reduction:.1f}%", ha='center', color='white', fontweight='bold')
        plt.savefig(os.path.join(SCRIPT_DIR, "graphs", "pipeline_size_breakdown.png"))
        print(f"Size breakdown graph saved to graphs/pipeline_size_breakdown.png")


def main():
    print("=== LiZIP Production Pipeline Benchmark (C++ Focused) ===")
    global TOTAL_FRAMES
    
    parser = argparse.ArgumentParser(description="Run Full Pipeline Benchmark")
    parser.add_argument('--dataset', type=str, default='nuscenes', choices=['nuscenes', 'kitti', 'argoverse'])
    parser.add_argument('--bin', type=str, default=MODEL_BIN, help='Path to .bin model')
    parser.add_argument('--mode', type=str, default='cpp', choices=['python', 'cpp', 'dual'], help='Benchmarking mode: python, cpp, or dual')
    parser.add_argument('--random', action='store_true', help='Use random sample of frames instead of first N')
    parser.add_argument('--frames', type=int, default=TOTAL_FRAMES)
    
    args = parser.parse_args()
    TOTAL_FRAMES = args.frames

    lidar_dir = get_lidar_dir(args.dataset)
    print(f"Using LiDAR directory: {lidar_dir}")
    
    encoder.device = device
    decoder.device = device
    
    model = None
    if args.mode in ['python', 'dual']:
        print(f"Loading Python Model for comparison: {MODEL_PATH}")
        parts = os.path.basename(args.bin).split('_')
        k_val = int(parts[1][1:]) if len(parts) > 1 else CONTEXT_SIZE
        h_val = int(parts[2][1:].split('.')[0]) if len(parts) > 2 else HIDDEN_DIM

        pth_path = args.bin.replace('.bin', '.pth')
        if not os.path.exists(pth_path):
            pth_path = MODEL_PATH
            
        model = PointPredictorMLP(context_size=k_val, hidden_dim=h_val)
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device); model.eval()

    files = encoding.get_files(lidar_dir, TOTAL_FRAMES, randomize=args.random)
    
    methods = []
    if args.mode in ['python', 'dual']:
        methods += ["LiZIP (Python, zlib)", "LiZIP (Python, lzma)"]
    if args.mode in ['cpp', 'dual']:
        methods += ["LiZIP (C++, zlib)", "LiZIP (C++, lzma)"]
        
    methods += ["Draco", "Laszip", "GZip"]
    
    enc_results = {k: [] for k in methods}
    dec_results = {k: [] for k in methods}
    errors = {k: [] for k in methods}
    sizes = {k: [] for k in methods}
    breakdowns = []
    
    ensure_dir(OUTPUT_DIR)

    for i, fpath in tqdm(enumerate(files), total=len(files), desc="Pipeline Progress", unit="frame"):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        gt_points_raw = load_point_cloud(fpath)[:, :3]
        
        # --- Python Implementation ---
        if "LiZIP (Python, zlib)" in methods:
            # zlib
            p_py_z = os.path.join(OUTPUT_DIR, f"{fname}.py.zlib.lizip")
            t_enc = encoding.encode_lizip(model, fpath, p_py_z, compression='zlib')
            enc_results["LiZIP (Python, zlib)"].append(t_enc)
            t_dec, pts = decoding.decode_lizip(model, p_py_z)
            dec_results["LiZIP (Python, zlib)"].append(t_dec)
            sizes["LiZIP (Python, zlib)"].append(os.path.getsize(p_py_z) / 1024.0)
            errors["LiZIP (Python, zlib)"].append(calculate_max_error(gt_points_raw, pts))
            os.remove(p_py_z)
            
        if "LiZIP (Python, lzma)" in methods:
            # lzma
            p_py_l = os.path.join(OUTPUT_DIR, f"{fname}.py.lzma.lizip")
            t_enc = encoding.encode_lizip(model, fpath, p_py_l, compression='lzma')
            enc_results["LiZIP (Python, lzma)"].append(t_enc)
            t_dec, pts = decoding.decode_lizip(model, p_py_l)
            dec_results["LiZIP (Python, lzma)"].append(t_dec)
            sizes["LiZIP (Python, lzma)"].append(os.path.getsize(p_py_l) / 1024.0)
            errors["LiZIP (Python, lzma)"].append(calculate_max_error(gt_points_raw, pts))
            os.remove(p_py_l)

        # --- C++ Implementation ---
        if "LiZIP (C++, zlib)" in methods:
            # zlib
            p_cpp_z = os.path.join(OUTPUT_DIR, f"{fname}.cpp.zlib.lizip")
            t_enc, _ = run_cpp_lizip("e", fpath, p_cpp_z, args.bin, compression="zlib")
            enc_results["LiZIP (C++, zlib)"].append(t_enc)
            p_rec = p_cpp_z + ".rec.bin"
            t_dec, _ = run_cpp_lizip("d", p_cpp_z, p_rec, args.bin)
            dec_results["LiZIP (C++, zlib)"].append(t_dec)
            sizes["LiZIP (C++, zlib)"].append(os.path.getsize(p_cpp_z) / 1024.0)
            if os.path.exists(p_rec):
                rec_pts = np.fromfile(p_rec, dtype=np.float32).reshape((-1, 3))
                errors["LiZIP (C++, zlib)"].append(calculate_max_error(gt_points_raw, rec_pts))
                os.remove(p_rec)
            os.remove(p_cpp_z)

        if "LiZIP (C++, lzma)" in methods:
            # lzma
            p_cpp_l = os.path.join(OUTPUT_DIR, f"{fname}.cpp.lzma.lizip")
            t_enc, b_stats = run_cpp_lizip("e", fpath, p_cpp_l, args.bin, compression="lzma")
            enc_results["LiZIP (C++, lzma)"].append(t_enc)
            breakdowns.append(b_stats)
            
            p_rec = p_cpp_l + ".rec.bin"
            t_dec, _ = run_cpp_lizip("d", p_cpp_l, p_rec, args.bin)
            dec_results["LiZIP (C++, lzma)"].append(t_dec)
            sizes["LiZIP (C++, lzma)"].append(os.path.getsize(p_cpp_l) / 1024.0)
            if os.path.exists(p_rec):
                rec_pts = np.fromfile(p_rec, dtype=np.float32).reshape((-1, 3))
                errors["LiZIP (C++, lzma)"].append(calculate_max_error(gt_points_raw, rec_pts))
                os.remove(p_rec)
            os.remove(p_cpp_l)

        # --- Baselines ---
        # Draco
        p_d = os.path.join(OUTPUT_DIR, f"{fname}.drc")
        t_enc = encoding.encode_draco(fpath, p_d)
        enc_results["Draco"].append(t_enc)
        t_dec, pts = decoding.decode_draco(p_d)
        dec_results["Draco"].append(t_dec)
        sizes["Draco"].append(os.path.getsize(p_d) / 1024.0)
        errors["Draco"].append(calculate_max_error(gt_points_raw, pts))
        os.remove(p_d)

        # Laszip
        p_l = os.path.join(OUTPUT_DIR, f"{fname}.laz")
        t_enc = encoding.encode_laszip(fpath, p_l)
        enc_results["Laszip"].append(t_enc)
        t_dec, pts = decoding.decode_laszip(p_l)
        dec_results["Laszip"].append(t_dec)
        sizes["Laszip"].append(os.path.getsize(p_l) / 1024.0)
        errors["Laszip"].append(calculate_max_error(gt_points_raw, pts))
        os.remove(p_l)

        # GZip
        p_g = os.path.join(OUTPUT_DIR, f"{fname}.gz")
        t_enc = encoding.encode_gzip(fpath, p_g)
        enc_results["GZip"].append(t_enc)
        t_dec, pts = decoding.decode_gzip(p_g)
        dec_results["GZip"].append(t_dec)
        sizes["GZip"].append(os.path.getsize(p_g) / 1024.0)
        errors["GZip"].append(calculate_max_error(gt_points_raw, pts))
        os.remove(p_g)

    plot_pipeline_results(enc_results, dec_results, errors, sizes, dataset_name=args.dataset, breakdown_stats=breakdowns)
    
    print("\nFinal Performance Summary (100% of frames):")
    print("-" * 115)
    print(f"{ 'Method':<25} | {'Total Time (s)':<20} | {'Size (KB)':<20} | {'Max Error (mm)':<20}")
    print("-" * 115)
    for m in methods:
        tot = np.array(enc_results[m]) + np.array(dec_results[m])
        print(f"{m:<25} | {np.mean(tot):.4f} +/- {np.std(tot):.4f}  | {np.mean(sizes[m]):.2f} +/- {np.std(sizes[m]):.2f}    | {np.mean(errors[m]):.4f}")
    print("-" * 115)

if __name__ == "__main__":
    main()
