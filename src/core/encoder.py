import torch
import numpy as np
import struct
import os
import time
import zlib
import lzma

from model import PointPredictorMLP
from data_loader import load_point_cloud
from voxel_sort import voxel_quantize_and_sort

MODEL_PATH = "models/mlp_v1.pth"
INPUT_FILE = "data/nuScenes/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604247644.pcd.bin"
OUTPUT_FILE = "data/compressed.lizip"
DEFAULT_DEBUG_GT_FILE = "data/debug_sorted_gt.bin"

CONTEXT_SIZE = 5 
INPUT_CHANNELS = CONTEXT_SIZE * 3 
BLOCK_SIZE = 128
RESID_SCALE = 100000.0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def shuffle_bytes(arr):
    """
    Rearranges bytes to improve compression.
    [A0 A1 A2 A3, B0 B1 B2 B3] -> [A0 B0, A1 B1, A2 B2, A3 B3]
    """
    n = arr.size
    # Ensure little-endian int32, view as (N, 4)
    arr_bytes = arr.astype('<i4').view(np.uint8).reshape(n, 4)
    # Transpose to (4, N) and flatten
    return arr_bytes.T.tobytes()

def encode_file_closed_loop(input_path, output_path, model, debug_gt_path=None, compression='zlib', input_format='auto'):
    print(f"Encoding (Zero-Drift Quantized, {compression}): {input_path}")
    
    # Dynamically detect context size from model
    first_layer = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][0]
    context_size = first_layer.in_features // 3
    print(f"Detected Model Context Size: {context_size}")

    if debug_gt_path is None:
        debug_gt_path = DEFAULT_DEBUG_GT_FILE

    # Load data
    raw_points = load_point_cloud(input_path)
    points_xyz = raw_points[:, :3]
    sorted_points = voxel_quantize_and_sort(points_xyz).astype(np.float32)
    
    points_mm = np.round(sorted_points * RESID_SCALE).astype(np.int32)
    
    os.makedirs(os.path.dirname(debug_gt_path) or '.', exist_ok=True)
    (points_mm.astype(np.float32) / RESID_SCALE).tofile(debug_gt_path)
    
    num_points = len(sorted_points)
    num_blocks = (num_points + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    points_mm_gpu = torch.from_numpy(points_mm).to(device)
    
    block_starts = np.arange(num_blocks) * BLOCK_SIZE
    heads_mm = []
    for start in block_starts:
        end = min(start + context_size, num_points)
        h = points_mm_gpu[start:end]
        if len(h) < context_size:
            padding = torch.zeros((context_size - len(h), 3), dtype=torch.int32, device=device)
            h = torch.cat((h, padding), dim=0)
        heads_mm.append(h)
    
    heads_mm_tensor = torch.stack(heads_mm)
    
    context = heads_mm_tensor.float() / RESID_SCALE
    context = context.reshape(num_blocks, -1)
    
    max_preds_per_block = BLOCK_SIZE - context_size
    residuals_tensor = torch.zeros((num_blocks, max_preds_per_block, 3), dtype=torch.int32, device=device)
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(max_preds_per_block):
            target_indices = block_starts + context_size + i
            valid_mask = target_indices < num_points
            
            if not any(valid_mask):
                break
                
            predictions_float = model(context)
            
            if torch.isnan(predictions_float).any():
                print(f"\nCRITICAL: NaN detected at step {i}")
                break

            targets_mm = torch.zeros((num_blocks, 3), dtype=torch.int32, device=device)
            targets_mm[valid_mask] = points_mm_gpu[target_indices[valid_mask]]
            
            preds_mm = torch.round(predictions_float * RESID_SCALE).int()
            resid_mm = targets_mm - preds_mm
            
            residuals_tensor[:, i, :] = resid_mm
            
            recon_mm = preds_mm + resid_mm
            recon_float = recon_mm.float() / RESID_SCALE
            
            context[:, :-3] = context[:, 3:]
            context[:, -3:] = recon_float

    print(f"Neural Prediction Complete. Time: {time.time() - start_time:.2f}s")

    valid_mask_all = torch.zeros((num_blocks, max_preds_per_block), dtype=torch.bool, device=device)

    bs_tensor = torch.from_numpy(block_starts).to(device).unsqueeze(1) # (B, 1)
    steps_tensor = torch.arange(max_preds_per_block, device=device).unsqueeze(0) # (1, P)
    target_indices_map = bs_tensor + context_size + steps_tensor # (B, P)
    valid_mask_all = target_indices_map < num_points
    
    valid_residuals = residuals_tensor[valid_mask_all] # (TotalValidPreds, 3)
    valid_residuals_np = valid_residuals.cpu().numpy()

    residuals_np = residuals_tensor.cpu().numpy()
    heads_np = heads_mm_tensor.cpu().numpy()
    
    # Delta Encode Heads to improve compression
    # shape: (num_blocks, context, 3) -> flat
    heads_flat = heads_np.reshape(-1, 3)
    heads_delta = np.zeros_like(heads_flat)
    heads_delta[0] = heads_flat[0]
    heads_delta[1:] = heads_flat[1:] - heads_flat[:-1]
    
    heads_shuffled = shuffle_bytes(heads_delta)
    residuals_shuffled = shuffle_bytes(residuals_np)
    
    # Payload Construction
    payload = heads_shuffled + residuals_shuffled
    
    # Entropy Compression
    comp_start = time.time()
    if compression == 'zlib':
        compressed_payload = zlib.compress(payload, level=9)
        comp_id = 1
    elif compression == 'lzma':
        compressed_payload = lzma.compress(payload, preset=9)
        comp_id = 2
    else:
        compressed_payload = payload
        comp_id = 0
    print(f"Entropy Compression ({compression}) Complete. Time: {time.time() - comp_start:.2f}s")
        
    # Header Construction (24 bytes)
    # Magic (4), CompID (1), Reserved (3), NumPoints (I), NumBlocks (I), Scale (f), TypeFlag (I)
    # Update Reserved to store Context Size in byte 1 of reserved field
    magic = b'LIZP'
    reserved = struct.pack('BBB', context_size, 0, 0)
    type_flag = 3 # Shuffled Int32
    
    header = struct.pack('<4sB3sIIfI', magic, comp_id, reserved, num_points, num_blocks, RESID_SCALE, type_flag)

    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(compressed_payload)
        
    return valid_residuals_np, sorted_points

if __name__ == "__main__":
    model = PointPredictorMLP(context_size=CONTEXT_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    
    encode_file_closed_loop(INPUT_FILE, OUTPUT_FILE, model)
