// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_THREADS = BLOCK_SIZE * GRID_SIZE;
const int MEM_SIZE = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
const int L2_CACHE_SIZE = 4 * 1024 * 1024; // 4MB L2 cache

// Kernel function to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(i * 3.14159f);
            result += cosf(i * 3.14159f);
            result += expf(i * 3.14159f);
            result += logf(i * 3.14159f);
        }
        data[idx] = result;
    }
}

// Kernel function to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += data[idx + i * BLOCK_SIZE];
        }
        data[idx] = result;
    }
}

// Kernel function to stress schedulers
__global__ void stress_scheduler(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            __syncwarp();
            result += data[idx + i * BLOCK_SIZE];
        }
        data[idx] = result;
    }
}

// Kernel function to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += __sinf(i * 3.14159f);
            result += __cosf(i * 3.14159f);
            result += __expf(i * 3.14159f);
            result += __logf(i * 3.14159f);
        }
        data[idx] = result;
    }
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&data[idx], 1.0f);
        }
    }
}

int main(int argc, char **argv) {
    int test_duration = 60; // default test duration in seconds
    if (argc > 1) {
        test_duration = std::stoi(argv[1]);
    }

    // Allocate memory on the GPU
    float *data;
    cudaMalloc((void **)&data, MEM_SIZE);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        stress_compute<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_memory<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_scheduler<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_xu<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_atomic<<<GRID_SIZE, BLOCK_SIZE>>>(data);

        // Check if test duration has been exceeded
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (elapsed_time > test_duration) {
            break;
        }
    }

    // Free memory
    cudaFree(data);

    return 0;
}