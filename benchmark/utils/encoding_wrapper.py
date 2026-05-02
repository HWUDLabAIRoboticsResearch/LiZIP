import os
import time
import gzip
import torch
from suppress import suppress_stdout
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.data_loader import load_point_cloud
from src.python import encoder

try:
    import DracoPy
except ImportError:
    DracoPy = None

try:
    import laspy
except ImportError:
    laspy = None

def encode_lizip(model, input_path, output_path, compression='zlib', gt_out_path=None):
    """Wraps the LiZIP Python encoder and returns execution time."""
    start_t = time.time()
    with suppress_stdout():
        encoder.encode_file_closed_loop(input_path, output_path, model, compression=compression, debug_gt_path=gt_out_path)
    return time.time() - start_t

def encode_draco(input_path, output_path, quantization_bits=24):
    """Wraps Google Draco encoder."""
    if DracoPy is None: 
        return 0
    
    # Draco expects (N, 3)
    points = load_point_cloud(input_path)[:, :3]
    
    start_t = time.time()
    # quantization_bits=24 is near-lossless for LiDAR scales
    drc_data = DracoPy.encode(points.flatten(), quantization_bits=quantization_bits, compression_level=1, create_metadata=False, preserve_order=True)
    with open(output_path, 'wb') as f:
        f.write(drc_data)
    return time.time() - start_t

def encode_laszip(input_path, output_path, scales=[1e-5, 1e-5, 1e-5]):
    """Wraps LASzip (via laspy)."""
    if laspy is None: 
        return 0
        
    try:
        points = load_point_cloud(input_path)
        
        start_t = time.time()
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = scales
        header.offsets = [0, 0, 0]
        las = laspy.LasData(header)
        las.x, las.y, las.z = points[:,0], points[:,1], points[:,2]
        las.write(output_path)
        return time.time() - start_t
    except Exception as e:
        print(f"Laszip Encode Error: {e}")
        return 0

def encode_gzip(input_path, output_path):
    """Standard GZip baseline (raw bytes)."""
    points = load_point_cloud(input_path)
    
    start_t = time.time()
    with gzip.open(output_path, 'wb') as f:
        f.write(points.tobytes())
    return time.time() - start_t

def get_files(lidar_dir, count=100, randomize=False):
    """Utility to gather files for benchmarking."""
    import glob
    import random
    
    patterns = ["*.bin", "*.pcd", "*.txt", "*.ply"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(lidar_dir, p)))
    
    files = sorted(list(set(files))) # Unique sorted list
    
    if randomize:
        return random.sample(files, min(len(files), count))
    
    return files[:count]
