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
const int TEST_DURATION = 60; // seconds

// Kernel to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sinf(data[idx] * i) * cosf(data[idx] * i);
    }
    data[idx] = result;
}

// Kernel to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[(idx + i) % MATRIX_SIZE];
    }
    data[idx] = result;
}

// Kernel to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += expf(data[idx] * i) * logf(data[idx] * i);
    }
    data[idx] = result;
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[0], 1.0f);
        result += data[idx];
    }
    data[idx] = result;
}

int main() {
    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // Allocate host memory
    float *h_data;
    cudaMallocHost((void **)&h_data, MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data;
    cudaMalloc((void **)&d_data, MATRIX_SIZE * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_data, h_data, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (true) {
        stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(d_data);
        stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(d_data);
        stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(d_data);
        stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_data);

        // Check test duration
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (milliseconds / 1000 > TEST_DURATION) {
            break;
        }
    }

    // Clean up
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}