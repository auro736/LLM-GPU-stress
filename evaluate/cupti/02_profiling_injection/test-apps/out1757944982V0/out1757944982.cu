// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

// Define constants
const int MATRIX_SIZE = 1024;
const int NUM_THREADS = 256;
const int NUM_BLOCKS = 256;
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
    __shared__ float shared_data[256];
    for (int i = 0; i < 1000; i++) {
        shared_data[threadIdx.x] = data[idx];
        __syncthreads();
        result += shared_data[threadIdx.x];
    }
    data[idx] = result;
}

// Kernel to stress schedulers
__global__ void scheduler_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[idx], 1.0f);
        result += data[idx];
    }
    data[idx] = result;
}

int main() {
    // Initialize CUDA device
    int device = 0;
    cudaSetDevice(device);

    // Allocate memory
    float *data;
    cudaMalloc((void **)&data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        compute_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        memory_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        scheduler_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data);
    }

    // Free memory
    cudaFree(data);

    return 0;
}