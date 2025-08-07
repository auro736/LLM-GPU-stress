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
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_ITERATIONS = 1000;

// Define kernel to stress computational units
__global__ void stress_compute(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 100; i++) {
            val = val * 2.0f + 3.0f * sinf(val);
        }
        matrix[idx] = val;
    }
}

// Define kernel to stress memory
__global__ void stress_memory(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 100; i++) {
            val = val + matrix[(idx + i) % (MATRIX_SIZE * MATRIX_SIZE)];
        }
        matrix[idx] = val;
    }
}

// Define kernel to stress XU units
__global__ void stress_xu(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 100; i++) {
            val = __expf(val);
        }
        matrix[idx] = val;
    }
}

// Define kernel to stress atomic operations
__global__ void stress_atomic(float* matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 100; i++) {
            atomicAdd(&matrix[idx], val);
        }
    }
}

int main(int argc, char** argv) {
    int test_duration = std::stoi(argv[1]);

    // Allocate host memory
    float* h_matrix;
    cudaMallocHost((void**)&h_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_matrix[i] = i * 1.0f;
    }

    // Allocate device memory
    float* d_matrix;
    cudaMalloc((void**)&d_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_matrix, h_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < test_duration) {
        stress_compute<<<GRID_SIZE, BLOCK_SIZE>>>(d_matrix);
        stress_memory<<<GRID_SIZE, BLOCK_SIZE>>>(d_matrix);
        stress_xu<<<GRID_SIZE, BLOCK_SIZE>>>(d_matrix);
        stress_atomic<<<GRID_SIZE, BLOCK_SIZE>>>(d_matrix);
    }

    // Copy device memory to host memory
    cudaMemcpy(h_matrix, d_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);

    // Free host memory
    cudaFreeHost(h_matrix);

    return 0;
}