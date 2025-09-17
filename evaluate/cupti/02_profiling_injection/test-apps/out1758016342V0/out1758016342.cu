// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define constants
#define MATRIX_SIZE 1024
#define TEST_DURATION 60 // seconds
#define NUM_STREAMS 16

// Kernel to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(idx * 0.01f) * cosf(idx * 0.01f);
        }
        data[idx] = result;
    }
}

// Kernel to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += data[idx] * data[idx];
        }
        data[idx] = result;
    }
}

// Kernel to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += __sinf(idx * 0.01f) * __cosf(idx * 0.01f);
        }
        data[idx] = result;
    }
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&data[idx], 1.0f);
        }
    }
}

int main() {
    // Initialize CUDA
    cudaDeviceReset();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    cudaSetDevice(0);

    // Allocate memory
    float *data;
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    cudaMalloc((void **)&data, size);

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize((MATRIX_SIZE * MATRIX_SIZE + blockSize.x - 1) / blockSize.x);
    clock_t start = clock();
    while ((clock() - start) / (double)CLOCKS_PER_SEC < TEST_DURATION) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            stress_compute<<<gridSize, blockSize, 0, streams[i]>>>(data);
            stress_memory<<<gridSize, blockSize, 0, streams[i]>>>(data);
            stress_xu<<<gridSize, blockSize, 0, streams[i]>>>(data);
            stress_atomic<<<gridSize, blockSize, 0, streams[i]>>>(data);
        }
    }

    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // Free memory
    cudaFree(data);

    return 0;
}