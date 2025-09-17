// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int MATRIX_SIZE = 1024;
const int NUM_THREADS = 256;
const int NUM_BLOCKS = 256;
const int TEST_DURATION = 60; // seconds

// Kernel function to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sinf(idx * 0.01f) * cosf(idx * 0.01f);
    }
    data[idx] = result;
}

// Kernel function to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_data[NUM_THREADS];
    shared_data[threadIdx.x] = data[idx];
    __syncthreads();
    data[idx] = shared_data[threadIdx.x] * 2.0f;
}

// Kernel function to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += __sinf(idx * 0.01f) * __cosf(idx * 0.01f);
    }
    data[idx] = result;
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[0], 1.0f);
    }
}

int main() {
    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // Allocate memory
    float *data;
    cudaMalloc((void **)&data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        cudaDeviceSynchronize();
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }
    }

    // Free memory
    cudaFree(data);

    return 0;
}