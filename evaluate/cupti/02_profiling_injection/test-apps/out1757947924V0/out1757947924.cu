// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// Define constants
const int MATRIX_SIZE = 1024;
const int BLOCK_SIZE = 16;
const int NUM_BLOCKS = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
const int TEST_DURATION = 60; // seconds

// Kernel to stress computational units
__global__ void compute_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(data[idx] * i) * cosf(data[idx] * i);
        }
        data[idx] = result;
    }
}

// Kernel to stress memory
__global__ void memory_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += data[(idx + i) % (MATRIX_SIZE * MATRIX_SIZE)];
        }
        data[idx] = result;
    }
}

// Kernel to stress XU units
__global__ void xu_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += __sinf(data[idx] * i) * __cosf(data[idx] * i);
        }
        data[idx] = result;
    }
}

// Kernel to stress atomic operations
__global__ void atomic_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&data[idx], 1.0f);
        }
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

    // Set CUDA device
    cudaSetDevice(0);

    // Allocate host memory
    float *h_data;
    cudaMallocHost((void **)&h_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data;
    cudaMalloc((void **)&d_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_data, h_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (true) {
        compute_stress<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data);
        memory_stress<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data);
        xu_stress<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data);
        atomic_stress<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data);

        // Check test duration
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (milliseconds / 1000.0f > TEST_DURATION) {
            break;
        }
    }

    // Clean up
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}