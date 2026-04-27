import numpy as np

def voxel_quantize_and_sort(points, grid_size=0.10):
    """
    Sorts points spatially based on a voxel grid.
    
    Args:
        points: (N, 4) numpy array [x, y, z, intensity]
        grid_size: The size of the voxel edge (e.g., 10cm)
        
    Returns:
        sorted_points: The re-ordered point cloud
    """
    voxel_coords = np.floor(points[:, :3] / grid_size).astype(np.int32)
    
    x = voxel_coords[:, 0]
    y = voxel_coords[:, 1]
    z = voxel_coords[:, 2]
    
    sort_indices = np.lexsort((x, y, z))
    
    sorted_points = points[sort_indices]
    
    return sorted_points