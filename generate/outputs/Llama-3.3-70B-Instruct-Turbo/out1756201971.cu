// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define matrix dimensions
#define MATRIX_DIM 1024

// Define block dimensions
#define BLOCK_DIM 16

// Define number of iterations
#define NUM_ITERATIONS 1000

// Kernel to stress computational units
__global__ void stress_compute(float *matrix1, float *matrix2, float *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < MATRIX_DIM && col < MATRIX_DIM) {
        float sum = 0.0f;
        for (int i = 0; i < MATRIX_DIM; i++) {
            sum += matrix1[row * MATRIX_DIM + i] * matrix2[i * MATRIX_DIM + col];
        }
        result[row * MATRIX_DIM + col] = sum;
    }
}

// Kernel to stress memory
__global__ void stress_memory(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_DIM * MATRIX_DIM) {
        array[idx] = array[idx] * 2.0f;
    }
}

// Kernel to stress XU units
__global__ void stress_xu(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_DIM * MATRIX_DIM) {
        array[idx] = sinf(array[idx]);
    }
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_DIM * MATRIX_DIM) {
        atomicAdd(&array[idx], 1.0f);
    }
}

int main() {
    // Initialize CUDA devices
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }

    // Allocate memory
    float *matrix1, *matrix2, *result, *array;
    size_t size = MATRIX_DIM * MATRIX_DIM * sizeof(float);
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&matrix1, size);
        cudaMalloc((void **)&matrix2, size);
        cudaMalloc((void **)&result, size);
        cudaMalloc((void **)&array, size);
    }

    // Initialize memory
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaMemset(matrix1, 1, size);
        cudaMemset(matrix2, 2, size);
        cudaMemset(result, 0, size);
        cudaMemset(array, 3, size);
    }

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < 120) {
        for (int i = 0; i < num_devices; i++) {
            cudaSetDevice(i);
            dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
            dim3 grid_dim((MATRIX_DIM + BLOCK_DIM - 1) / BLOCK_DIM, (MATRIX_DIM + BLOCK_DIM - 1) / BLOCK_DIM);
            stress_compute<<<grid_dim, block_dim>>>(matrix1, matrix2, result);
            stress_memory<<<grid_dim, block_dim>>>(array);
            stress_xu<<<grid_dim, block_dim>>>(array);
            stress_atomic<<<grid_dim, block_dim>>>(array);
        }
    }

    // Clean up
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaFree(matrix1);
        cudaFree(matrix2);
        cudaFree(result);
        cudaFree(array);
    }

    return 0;
}