// stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1024
#define TEST_DURATION 60 // seconds
#define NUM_STREAMS 16

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
            result += __expf(data[idx] * i) * __logf(data[idx] * i);
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
    // Initialize CUDA
    cudaDeviceReset();
    int device;
    cudaGetDevice(&device);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate memory
    float *data;
    cudaMalloc((void **)&data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize data
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }

    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize((MATRIX_SIZE * MATRIX_SIZE + blockSize.x - 1) / blockSize.x);
    clock_t start = clock();
    while ((clock() - start) / (double)CLOCKS_PER_SEC < TEST_DURATION) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            compute_stress<<<gridSize, blockSize, 0, streams[i]>>>(data);
            memory_stress<<<gridSize, blockSize, 0, streams[i]>>>(data);
            xu_stress<<<gridSize, blockSize, 0, streams[i]>>>(data);
            atomic_stress<<<gridSize, blockSize, 0, streams[i]>>>(data);
        }
    }

    // Clean up
    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(data);
    cudaDeviceReset();

    return 0;
}