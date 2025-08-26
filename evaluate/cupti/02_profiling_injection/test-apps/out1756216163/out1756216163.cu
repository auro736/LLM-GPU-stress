// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <chrono>
#include <thread>

#define MATRIX_SIZE 1024
#define NUM_BLOCKS 16
#define NUM_THREADS 256
#define TEST_DURATION 120 // seconds

// Kernel to stress computational units
__global__ void stress_compute(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(idx * 0.01f) * cosf(idx * 0.01f);
        }
        matrix[idx] = result;
    }
}

!__global__ void stress_memory(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += matrix[idx] * matrix[idx];
        }
        matrix[idx] = result;
    }
}

// Kernel to stress XU units
__global__ void stress_xu(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += __sinf(idx * 0.01f) * __cosf(idx * 0.01f);
        }
        matrix[idx] = result;
    }
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&matrix[idx], 1.0f);
        }
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

    // Allocate memory on each device
    float *matrix[2];
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&matrix[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    }

    // Launch kernels on each device
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(i);
            stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
            stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
            stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
            stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
        }
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }
    }

    // Free memory on each device
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaFree(matrix[i]);
    }

    return 0;
}