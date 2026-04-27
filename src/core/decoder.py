import torch
import numpy as np
import struct
import time
import os
import zlib
import lzma

from model import PointPredictorMLP
from data_loader import save_kitti_data

MODEL_PATH = "models/mlp_v1.pth" 
COMPRESSED_FILE = "data/compressed.lizip"
RECONSTRUCTED_FILE = "data/reconstructed.bin"
CONTEXT_SIZE = 5 
INPUT_CHANNELS = CONTEXT_SIZE * 3
BLOCK_SIZE = 128

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def unshuffle_bytes(data_bytes, n):
    """
    Reverses shuffle_bytes.
    """
    arr_shuffled = np.frombuffer(data_bytes, dtype=np.uint8).reshape(4, n)
    arr_bytes = np.ascontiguousarray(arr_shuffled.T)
    return arr_bytes.view('<i4').flatten()

def decode_file(input_path, output_path, model, output_format='auto'):
    print(f"Decoding (Zero-Drift Quantized): {input_path}")
    start_time = time.time()
    
    with open(input_path, 'rb') as f:
        header_bytes = f.read(24)
        if len(header_bytes) < 24:
            f.seek(0)
            header_bytes = f.read(16)
            num_points, num_blocks, scale, type_flag = struct.unpack('<IIfI', header_bytes)
            comp_id = 0
            print("Warning: Old file format detected.")
            payload_compressed = f.read()
        else:
            magic, comp_id, reserved, num_points, num_blocks, scale, type_flag = struct.unpack('<4sB3sIIfI', header_bytes)
            if magic != b'LIZP':
                f.seek(0)
                header_bytes = f.read(16)
                num_points, num_blocks, scale, type_flag = struct.unpack('<IIfI', header_bytes)
                comp_id = 0
                print("Warning: Old file format detected (No Magic).")
                f.seek(16)
            payload_compressed = f.read()

    # Entropy Decompression
    decomp_start = time.time()
    if comp_id == 1:
        payload = zlib.decompress(payload_compressed)
    elif comp_id == 2:
        payload = lzma.decompress(payload_compressed)
    else:
        payload = payload_compressed
    print(f"Entropy Decompression Complete. Time: {time.time() - decomp_start:.2f}s")

    # Extract context size from reserved field (byte 0)
    context_size = reserved[0]
    if context_size == 0: context_size = 5 # Backwards compatibility
    print(f"Detected Context Size from Header: {context_size}")

    # Split payload into heads and residuals
    head_bytes_len = num_blocks * context_size * 3 * 4
    head_raw = payload[:head_bytes_len]
    residuals_bytes = payload[head_bytes_len:]

    if type_flag == 3:
        # Unshuffle heads
        heads_delta = unshuffle_bytes(head_raw, num_blocks * context_size * 3).reshape((-1, 3))
    else:
        heads_delta = np.frombuffer(head_raw, dtype=np.int32).reshape((-1, 3))
        
    heads_flat = np.cumsum(heads_delta, axis=0, dtype=np.int32)
    heads_mm = heads_flat.reshape((num_blocks, context_size, 3))
    
    # Residuals
    max_preds_per_block = BLOCK_SIZE - context_size
    
    if type_flag == 3:
        # Unshuffle residuals
        resids_mm = unshuffle_bytes(residuals_bytes, num_blocks * max_preds_per_block * 3)
        resids_mm = resids_mm.reshape((num_blocks, max_preds_per_block, 3))
    elif type_flag == 2:
        resids_mm = np.frombuffer(residuals_bytes, dtype=np.int32).reshape((num_blocks, max_preds_per_block, 3))
    else:
        resids_mm = np.frombuffer(residuals_bytes, dtype=np.int16).reshape((num_blocks, max_preds_per_block, 3))
    
    print(f"Reconstructing {num_points} points (Parallel Blocks, Scale: {scale}, Type: {type_flag})...")
    
    context = torch.tensor(heads_mm, device=device).float() / scale
    context = context.reshape(num_blocks, -1)
    
    resids_gpu = torch.tensor(resids_mm.astype(np.int32), device=device)
    
    reconstructed_mm = torch.zeros((num_blocks, BLOCK_SIZE, 3), dtype=torch.int32, device=device)
    reconstructed_mm[:, :context_size, :] = torch.tensor(heads_mm, device=device)
    
    with torch.no_grad():
        for i in range(max_preds_per_block):
            predictions_float = model(context)
            
            preds_mm = torch.round(predictions_float * scale).int()
            
            actual_mm = preds_mm + resids_gpu[:, i, :]
            reconstructed_mm[:, context_size + i, :] = actual_mm
            
            context[:, :-3] = context[:, 3:]
            context[:, -3:] = actual_mm.float() / scale

    print(f"Neural Decoding Complete. Time: {time.time() - start_time:.2f}s")
    
    final_cloud = reconstructed_mm.cpu().numpy().reshape(-1, 3).astype(np.float32) / scale
    final_cloud = final_cloud[:num_points]
    
    # Determine output format
    if output_format == 'auto':
        if output_path.endswith('.txt'):
            output_format = 'kitti'
        else:
            output_format = 'bin'

    if output_format == 'kitti':
        save_kitti_data(final_cloud, output_path)
    else:
        final_cloud.tofile(output_path)
        
    print(f"Saved to {output_path}")
    
    return final_cloud

if __name__ == "__main__":
    model = PointPredictorMLP(context_size=CONTEXT_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    
    if os.path.exists(COMPRESSED_FILE):
        decode_file(COMPRESSED_FILE, RECONSTRUCTED_FILE, model)
    else:
        print("Run encoder.py first!")