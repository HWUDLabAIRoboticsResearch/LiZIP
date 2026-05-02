import os
import time
import gzip
import numpy as np
from suppress import suppress_stdout
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.python import decoder

try:
    import DracoPy
except ImportError:
    DracoPy = None

try:
    import laspy
except ImportError:
    laspy = None

def decode_lizip(model, input_path):
    """Wraps LiZIP Python decoder and returns (time, points)."""
    start_t = time.time()
    try:
        rec_out = input_path + ".rec.bin"
        with suppress_stdout():
            decoder.decode_file(input_path, rec_out, model)
        
        dec_time = time.time() - start_t
        points = np.fromfile(rec_out, dtype=np.float32).reshape((-1, 3))
        
        if os.path.exists(rec_out): 
            os.remove(rec_out)
            
        return dec_time, points
    except Exception as e:
        print(f"LiZIP Decode Error: {e}")
        return 0, None

def decode_draco(input_path):
    """Wraps Google Draco decoder."""
    if DracoPy is None: 
        return 0, None
    try:
        start_t = time.time()
        with open(input_path, 'rb') as f:
            data = f.read()
        decoded_obj = DracoPy.decode(data)
        points = np.array(decoded_obj.points).astype(np.float32).reshape((-1, 3))
        return time.time() - start_t, points
    except Exception as e:
        print(f"Draco Decode Error: {e}")
        return 0, None

def decode_laszip(input_path):
    """Wraps LASzip (via laspy) decoder."""
    if laspy is None: 
        return 0, None
    try:
        start_t = time.time()
        with laspy.open(input_path) as f:
            las = f.read()
            points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        return time.time() - start_t, points
    except Exception as e:
        print(f"LASzip Decode Error: {e}")
        return 0, None

def decode_gzip(input_path):
    """Standard GZip decompression."""
    try:
        start_t = time.time()
        with gzip.open(input_path, 'rb') as f:
            data = f.read()
        
        # Determine shape based on data size
        raw_data = np.frombuffer(data, dtype=np.float32)
        if raw_data.size % 5 == 0:
            points = raw_data.reshape((-1, 5))[:, :3]
        elif raw_data.size % 4 == 0:
            points = raw_data.reshape((-1, 4))[:, :3]
        else:
            points = raw_data.reshape((-1, 3))
            
        return time.time() - start_t, points
    except Exception as e:
        print(f"GZip Decode Error: {e}")
        return 0, None
