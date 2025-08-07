// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1024
#define NUM_THREADS 256
#define NUM_BLOCKS 16
#define TEST_DURATION 60 // seconds

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
        result += data[idx + i * blockDim.x];
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
        atomicAdd(&data[idx], 1.0f);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <test_duration>\n", argv[0]);
        return 1;
    }

    int test_duration = atoi(argv[1]);
    if (test_duration <= 0) {
        printf("Test duration must be greater than 0\n");
        return 1;
    }

    // Initialize CUDA device
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using device: %s\n", prop.name);

    // Allocate memory
    float *data;
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    cudaMalloc((void **)&data, size);

    // Launch kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (1) {
        stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(data);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        if (elapsed_time > test_duration * 1000) {
            break;
        }
    }

    // Clean up
    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}