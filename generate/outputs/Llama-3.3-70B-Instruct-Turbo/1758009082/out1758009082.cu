// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int MATRIX_SIZE = 1024;
const int NUM_THREADS = 256;
const int NUM_BLOCKS = 256;
const int TEST_DURATION = 60; // seconds

// Kernel function to stress computational units
__global__ void stress_compute(float *matrix1, float *matrix2, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            sum += matrix1[idx * MATRIX_SIZE + i] * matrix2[i * MATRIX_SIZE + idx % MATRIX_SIZE];
        }
        result[idx] = sum;
    }
}

// Kernel function to stress memory
__global__ void stress_memory(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        array[idx] = array[idx] * 2.0f;
    }
}

// Kernel function to stress XU units
__global__ void stress_xu(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        array[idx] = __sinf(array[idx]);
    }
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        atomicAdd(&array[idx], 1.0f);
    }
}

int main() {
    // Allocate host memory
    float *h_matrix1, *h_matrix2, *h_result, *h_array;
    cudaMallocHost((void **)&h_matrix1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_matrix2, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_result, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_matrix1[i] = 1.0f;
        h_matrix2[i] = 2.0f;
        h_result[i] = 0.0f;
        h_array[i] = 1.0f;
    }

    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_result, *d_array;
    cudaMalloc((void **)&d_matrix1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_matrix2, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_result, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_matrix1, h_matrix1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(d_matrix1, d_matrix2, d_result);
        stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(d_array);
        stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(d_array);
        stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_array);
        cudaDeviceSynchronize();
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }
    }

    // Free host and device memory
    cudaFreeHost(h_matrix1);
    cudaFreeHost(h_matrix2);
    cudaFreeHost(h_result);
    cudaFreeHost(h_array);
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);
    cudaFree(d_array);

    return 0;
}