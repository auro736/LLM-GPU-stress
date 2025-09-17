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
            val = val * 2.0f + 1.0f;
            matrix[idx] = val;
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
    // Initialize CUDA device
    int device_id = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Using device: " << device_prop.name << std::endl;

    // Allocate memory
    float* h_matrix;
    float* d_matrix;
    size_t matrix_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    h_matrix = (float*)malloc(matrix_size);
    cudaMalloc((void**)&d_matrix, matrix_size);

    // Initialize matrix
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_matrix[i] = (float)i;
    }
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        stress_compute<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_matrix);
        stress_memory<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_matrix);
        stress_xu<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_matrix);
        stress_atomic<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_matrix);
        cudaDeviceSynchronize();
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }
    }

    // Clean up
    free(h_matrix);
    cudaFree(d_matrix);

    return 0;
}