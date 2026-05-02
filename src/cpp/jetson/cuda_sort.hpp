#ifndef CUDA_SORT_HPP
#define CUDA_SORT_HPP

#include <vector>

struct Point;

void gpu_voxel_sort(std::vector<Point>& points, float grid = 0.10f);

#endif
