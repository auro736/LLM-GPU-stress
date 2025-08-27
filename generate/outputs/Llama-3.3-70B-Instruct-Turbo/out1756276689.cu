// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>

// Define constants
const int MATRIX_SIZE = 1024;
const int NUM_BLOCKS = 16;
const int NUM_THREADS = 256;
const int TEST_DURATION = 120; // seconds

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
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx + i * blockDim.x];
    }
    data[idx] = result;
}

// Kernel function to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += expf(idx * 0.01f) * logf(idx * 0.01f);
    }
    data[idx] = result;
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[idx], 1.0f);
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
    float *data[2];
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&data[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    }

    // Start test
    clock_t start_time = clock();
    while ((clock() - start_time) / (double)CLOCKS_PER_SEC < TEST_DURATION) {
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(i);
            stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(data[i]);
            stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(data[i]);
            stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(data[i]);
            stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(data[i]);
        }
    }

    // Clean up
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaFree(data[i]);
    }

    return 0;
}