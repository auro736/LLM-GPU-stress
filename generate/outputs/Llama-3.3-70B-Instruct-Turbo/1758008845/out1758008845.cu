// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 16;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int TEST_DURATION = 60; // seconds

// Kernel function to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sinf(idx * 0.01f) * cosf(idx * 0.01f);
    }
    data[idx] = result;
}

// Kernel function to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx * i];
    }
    data[idx] = result;
}

// Kernel function to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += __sinf(idx * 0.01f) * __cosf(idx * 0.01f);
    }
    data[idx] = result;
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[idx], 0.01f);
    }
}

int main() {
    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // Allocate memory
    float *data;
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    cudaMalloc((void **)&data, size);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        stress_compute<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_memory<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_xu<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_atomic<<<GRID_SIZE, BLOCK_SIZE>>>(data);
    }

    // Free memory
    cudaFree(data);

    return 0;
}