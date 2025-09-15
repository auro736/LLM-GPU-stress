// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 256;
const int NUM_BLOCKS = 1024;
const int MATRIX_SIZE = 1024;
const int TEST_DURATION = 60; // seconds

// Kernel to stress computational units
__global__ void stress_compute(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            val = val * 2.0f + 1.0f;
        }
        matrix[idx] = val;
    }
}

// Kernel to stress memory
__global__ void stress_memory(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            matrix[idx] = val * 2.0f;
            val = matrix[idx];
        }
    }
}

// Kernel to stress XU units
__global__ void stress_xu(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            val = __sinf(val);
        }
        matrix[idx] = val;
    }
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&matrix[idx], 1.0f);
        }
    }
}

int main() {
    // Allocate memory on the GPU
    float* matrix;
    cudaMalloc((void**)&matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize matrix
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrix[i] = 1.0f;
    }

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        stress_compute<<<NUM_BLOCKS, BLOCK_SIZE>>>(matrix);
        stress_memory<<<NUM_BLOCKS, BLOCK_SIZE>>>(matrix);
        stress_xu<<<NUM_BLOCKS, BLOCK_SIZE>>>(matrix);
        stress_atomic<<<NUM_BLOCKS, BLOCK_SIZE>>>(matrix);
    }

    // Free memory
    cudaFree(matrix);

    return 0;
}