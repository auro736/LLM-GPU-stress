// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

#define MATRIX_SIZE 1024
#define NUM_BLOCKS 16
#define NUM_THREADS 256
#define TEST_DURATION 120 // seconds

// Kernel to stress computational units
__global__ void compute_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx] * data[idx];
    }
    data[idx] = result;
}

// Kernel to stress memory
__global__ void memory_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx + i];
    }
    data[idx] = result;
}

// Kernel to stress XU units
__global__ void xu_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sinf(data[idx]);
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
    int num_gpus = 2;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;

    // Allocate memory on each GPU
    float *data[2];
    cudaSetDevice(0);
    cudaMalloc((void **)&data[0], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaSetDevice(1);
    cudaMalloc((void **)&data[1], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize data
    cudaSetDevice(0);
    cudaMemset(data[0], 1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaSetDevice(1);
    cudaMemset(data[1], 1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        // Launch kernels on each GPU
        cudaSetDevice(0);
        compute_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[0]);
        memory_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[0]);
        xu_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[0]);
        atomic_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[0]);

        cudaSetDevice(1);
        compute_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[1]);
        memory_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[1]);
        xu_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[1]);
        atomic_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data[1]);

        // Check test duration
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }
    }

    // Free memory
    cudaSetDevice(0);
    cudaFree(data[0]);
    cudaSetDevice(1);
    cudaFree(data[1]);

    return 0;
}