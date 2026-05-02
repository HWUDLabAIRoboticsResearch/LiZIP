#include "cuda_sort.hpp"
#include "types.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <algorithm>

__device__ inline uint64_t part1by2_gpu(uint64_t n) {
    n &= 0x1fffff;
    n = (n | (n << 32)) & 0x1f00000000ffffULL;
    n = (n | (n << 16)) & 0x1f0000ff0000ffULL;
    n = (n | (n << 8))  & 0x100f00f00f00f00fULL;
    n = (n | (n << 4))  & 0x10c30c30c30c30c3ULL;
    n = (n | (n << 2))  & 0x1249249249249249ULL;
    return n;
}

__global__ void compute_morton_codes(const Point* points, uint64_t* codes, int n, 
                                     float min_x, float min_y, float min_z, float grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t vx = (uint64_t)((points[i].x - min_x) / grid);
    uint64_t vy = (uint64_t)((points[i].y - min_y) / grid);
    uint64_t vz = (uint64_t)((points[i].z - min_z) / grid);

    codes[i] = part1by2_gpu(vx) | (part1by2_gpu(vy) << 1) | (part1by2_gpu(vz) << 2);
}

void gpu_voxel_sort(std::vector<Point>& points, float grid) {
    if (points.empty()) return;

    int n = points.size();
    float min_x = points[0].x, min_y = points[0].y, min_z = points[0].z;
    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        min_z = std::min(min_z, p.z);
    }

    thrust::device_vector<Point> d_points = points;
    thrust::device_vector<uint64_t> d_codes(n);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    compute_morton_codes<<<blocks, threads>>>(thrust::raw_pointer_cast(d_points.data()), 
                                             thrust::raw_pointer_cast(d_codes.data()), 
                                             n, min_x, min_y, min_z, grid);

    thrust::sort_by_key(d_codes.begin(), d_codes.end(), d_points.begin());

    thrust::copy(d_points.begin(), d_points.end(), points.begin());
}
