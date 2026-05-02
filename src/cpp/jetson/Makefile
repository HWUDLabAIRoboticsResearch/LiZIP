# LiZIP C++ Engine Makefile for Jetson (AArch64)

CXX = g++
NVCC = /usr/local/cuda/bin/nvcc
CXXFLAGS = -O3 -fopenmp -std=c++17
NVCCFLAGS = -O3 -std=c++17 --generate-code arch=compute_87,code=sm_87 
INCLUDES = -I/usr/local/cuda/include -I/usr/include/aarch64-linux-gnu
LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/lib/aarch64-linux-gnu
LIBS = -lz -llzma -lcudart -lnvinfer

SRC_CPP = lizip.cpp
SRC_CU = cuda_sort.cu
OBJ_CU = cuda_sort.o
TARGET = lizip

all: $(TARGET)

$(OBJ_CU): $(SRC_CU) cuda_sort.hpp types.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $(SRC_CU) -o $(OBJ_CU)

$(TARGET): $(SRC_CPP) $(OBJ_CU) trt_predictor.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC_CPP) $(OBJ_CU) -o $(TARGET) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) $(OBJ_CU)
