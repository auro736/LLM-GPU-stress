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
        result /= data[idx] + 1.0f;
    }
    data[idx] = result;
}

// Kernel to stress memory
__global__ void memory_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx + i * blockDim.x];
    }
    data[idx] = result;
}

// Kernel to stress XU units
__global__ void xu_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sinf(data[idx]);
        result += cosf(data[idx]);
        result += expf(data[idx]);
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
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 2) {
        std::cerr << "At least two GPUs are required for this test." << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    for (int i = 0; i < num_gpus; i++) {
        cudaGetDeviceProperties(&prop, i);
        if (prop.totalGlobalMem < 48 * 1024 * 1024 * 1024) {
            std::cerr << "At least 48GB of VRAM is required for this test." << std::endl;
            return 1;
        }
    }

    float *data1, *data2;
    cudaMalloc((void **)&data1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&data2, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        compute_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data1);
        memory_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data2);
        xu_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data1);
        atomic_stress<<<NUM_BLOCKS, NUM_THREADS>>>(data2);
    }

    cudaFree(data1);
    cudaFree(data2);

    return 0;
}