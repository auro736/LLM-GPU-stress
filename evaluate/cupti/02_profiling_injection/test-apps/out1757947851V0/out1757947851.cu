// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int TEST_DURATION = 60; // seconds

// Kernel to stress computational units
__global__ void compute_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sinf(idx * 0.01f) * cosf(idx * 0.01f);
    }
    data[idx] = result;
}

// Kernel to stress memory
__global__ void memory_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx + i * GRID_SIZE * BLOCK_SIZE];
    }
    data[idx] = result;
}

// Kernel to stress XU units
__global__ void xu_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += __sinf(idx * 0.01f) * __cosf(idx * 0.01f);
    }
    data[idx] = result;
}

// Kernel to stress atomic operations
__global__ void atomic_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[idx], 1.0f);
    }
}

int main() {
    // Allocate memory
    float *d_data;
    cudaMalloc((void **)&d_data, GRID_SIZE * BLOCK_SIZE * sizeof(float));

    // Start test
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        // Launch kernels
        compute_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        memory_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        xu_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        atomic_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);

        // Check test duration
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }
    }

    // Clean up
    cudaFree(d_data);

    return 0;
}