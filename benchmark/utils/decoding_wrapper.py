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
import argparse
import decoder

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

ENCODED_DIR = os.path.join(SCRIPT_DIR, "data", "benchmark_out")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_extended_v2.pth")
CONTEXT_SIZE = 5
HIDDEN_DIM = 512
WARMUP_FRAMES = 5
TOTAL_FRAMES = 55

def get_lidar_dir(dataset_name):
    if dataset_name.lower() == 'kitti':
        return os.path.join(PROJECT_ROOT, "data", "KITTI")
    elif dataset_name.lower() == 'argoverse':
        return os.path.join(PROJECT_ROOT, "data", "argoverse")
    else:
        return os.path.join(PROJECT_ROOT, "data", "nuScenes", "LIDAR_TOP")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def decode_lizip(model, input_path):
    start_t = time.time()
    
    try:
        rec_out = input_path + ".rec.bin"
        
        with suppress_stdout():
            decoder.decode_file(input_path, rec_out, model)
        
        dec_time = time.time() - start_t
        
        points = np.fromfile(rec_out, dtype=np.float32).reshape((-1, 3))
        
        if os.path.exists(rec_out): os.remove(rec_out)
        
        return dec_time, points
        
    except Exception as e:
        print(f"LiZIP Decode Error {input_path}: {e}")
        return 0, None

def decode_draco(input_path):
    if DracoPy is None: return 0, None
    try:
        start_t = time.time()
        with open(input_path, 'rb') as f:
            data = f.read()
        decoded_obj = DracoPy.decode(data)
        points = np.array(decoded_obj.points).astype(np.float32)
        dec_time = time.time() - start_t
        return dec_time, points
    except Exception as e:
        print(f"Draco Decode Error {input_path}: {e}")
        return 0, None

def decode_laszip(input_path):
    if laspy is None: return 0, None
    try:
        start_t = time.time()
        with laspy.open(input_path) as f:
            las = f.read()
            x = las.x
            y = las.y
            z = las.z
            points = np.vstack((x, y, z)).T.astype(np.float32)
        dec_time = time.time() - start_t
        return dec_time, points
    except Exception as e:
        print(f"LASzip Decode Error {input_path}: {e}")
        return 0, None

def decode_gzip(input_path):
    try:
        start_t = time.time()
        with gzip.open(input_path, 'rb') as f:
            data = f.read()
        
        # Try reshaping to (N, 4) first (NuScenes/KITTI)
        try:
            points = np.frombuffer(data, dtype=np.float32).reshape((-1, 4))
        except ValueError:
            # Try (N, 5)
            try:
                points = np.frombuffer(data, dtype=np.float32).reshape((-1, 5))
            except ValueError:
                # Try (N, 3) (Argoverse/PLY)
                points = np.frombuffer(data, dtype=np.float32).reshape((-1, 3))
        
        dec_time = time.time() - start_t
        return dec_time, points[:, :3]
    except Exception as e:
        print(f"GZip Decode Error {input_path}: {e}")
        return 0, None

def benchmark_decoding(model, files, encoded_dir):
    results = {
        "LiZIP (Python, zlib)": [],
        "LiZIP (Python, lzma)": [],
        "Draco (Lossy)": [],
        "Draco (Lossless)": [],
        "LASzip (Lossy)": [],
        "LASzip (Lossless)": [],
        "GZip": []
    }
    
    reconstructions = {
        "LiZIP (Python, zlib)": [],
        "LiZIP (Python, lzma)": [],
        "Draco (Lossy)": [],
        "Draco (Lossless)": [],
        "LASzip (Lossy)": [],
        "LASzip (Lossless)": [],
        "GZip": []
    }
    
    print(f"Starting Decoding Benchmark on {len(files)} frames...")
    
    for i, fpath in tqdm(enumerate(files), total=len(files), desc="Decoding", unit="frame"):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        
        p_lizip_zlib = os.path.join(encoded_dir, f"{fname}.zlib.lizip")
        p_lizip_lzma = os.path.join(encoded_dir, f"{fname}.lzma.lizip")
        p_draco_lossy = os.path.join(encoded_dir, f"{fname}.lossy.drc")
        p_draco_lossless = os.path.join(encoded_dir, f"{fname}.lossless.drc")
        p_las_lossy   = os.path.join(encoded_dir, f"{fname}.lossy.laz")
        p_las_lossless   = os.path.join(encoded_dir, f"{fname}.lossless.laz")
        p_gzip  = os.path.join(encoded_dir, f"{fname}.gz")
        
        # LiZIP (zlib)
        if os.path.exists(p_lizip_zlib):
            t, pts = decode_lizip(model, p_lizip_zlib)
            results["LiZIP (Python, zlib)"].append(t)
            reconstructions["LiZIP (Python, zlib)"].append(pts)
        else:
            results["LiZIP (Python, zlib)"].append(0)
            reconstructions["LiZIP (Python, zlib)"].append(None)

        # LiZIP (lzma)
        if os.path.exists(p_lizip_lzma):
            t, pts = decode_lizip(model, p_lizip_lzma)
            results["LiZIP (Python, lzma)"].append(t)
            reconstructions["LiZIP (Python, lzma)"].append(pts)
        else:
            results["LiZIP (Python, lzma)"].append(0)
            reconstructions["LiZIP (Python, lzma)"].append(None)
            
        # Draco (Lossy)
        if os.path.exists(p_draco_lossy):
            t, pts = decode_draco(p_draco_lossy)
            results["Draco (Lossy)"].append(t)
            reconstructions["Draco (Lossy)"].append(pts)
        else:
            results["Draco (Lossy)"].append(0)
            reconstructions["Draco (Lossy)"].append(None)

        # Draco (Lossless)
        if os.path.exists(p_draco_lossless):
            t, pts = decode_draco(p_draco_lossless)
            results["Draco (Lossless)"].append(t)
            reconstructions["Draco (Lossless)"].append(pts)
        else:
            results["Draco (Lossless)"].append(0)
            reconstructions["Draco (Lossless)"].append(None)

        # Laszip (Lossy)
        if os.path.exists(p_las_lossy):
            t, pts = decode_laszip(p_las_lossy)
            results["LASzip (Lossy)"].append(t)
            reconstructions["LASzip (Lossy)"].append(pts)
        else:
            results["LASzip (Lossy)"].append(0)
            reconstructions["LASzip (Lossy)"].append(None)

        # Laszip (Lossless)
        if os.path.exists(p_las_lossless):
            t, pts = decode_laszip(p_las_lossless)
            results["LASzip (Lossless)"].append(t)
            reconstructions["LASzip (Lossless)"].append(pts)
        else:
            results["LASzip (Lossless)"].append(0)
            reconstructions["LASzip (Lossless)"].append(None)

        # GZip
        if os.path.exists(p_gzip):
            t, pts = decode_gzip(p_gzip)
            results["GZip"].append(t)
            reconstructions["GZip"].append(pts)
        else:
            results["GZip"].append(0)
            reconstructions["GZip"].append(None)

    return results, reconstructions

def plot_decoding_results(results, warmup=0, dataset_name="NuScenes"):
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
        
    plt.title(f"Decoding Time per Frame ({ds_display})")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_file = os.path.join(SCRIPT_DIR, "graphs", "decoding_benchmark.png")
    ensure_dir(os.path.dirname(out_file))
    plt.savefig(out_file)
    print(f"Decoding graph saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Decoding Benchmark")
    parser.add_argument('--dataset', type=str, default='nuscenes', choices=['nuscenes', 'kitti', 'argoverse'], help='Dataset to use (nuscenes, kitti, argoverse)')
    args = parser.parse_args()

    lidar_dir = get_lidar_dir(args.dataset)
    print(f"Using LiDAR directory: {lidar_dir}")
    
    decoder.device = device
    
    print(f"Loading model: {MODEL_PATH}")
    model = PointPredictorMLP(context_size=CONTEXT_SIZE, hidden_dim=HIDDEN_DIM)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    import glob
    files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))
    if not files:
        files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))
    if not files:
        files = sorted(glob.glob(os.path.join(lidar_dir, "*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    files = files[:TOTAL_FRAMES]
    
    results, _ = benchmark_decoding(model, files, ENCODED_DIR)
    plot_decoding_results(results, WARMUP_FRAMES)
    
    print("\nDecoding Stats (Mean +/- Std) [Frames 6-55]:")
    for k, v in results.items():
        if len(v) > WARMUP_FRAMES:
            valid = v[WARMUP_FRAMES:]
            print(f"{k:<10}: {np.mean(valid):.4f} +/- {np.std(valid):.4f} s")

if __name__ == "__main__":
    main()
