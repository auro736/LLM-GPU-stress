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
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sinf(data[idx] * 3.14159f);
        result += cosf(data[idx] * 3.14159f);
        result += expf(data[idx]);
        result += logf(data[idx] + 1.0f);
    }
    data[idx] = result;
}

// Kernel to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx + i * MATRIX_SIZE];
        result += data[idx + i * MATRIX_SIZE + 1];
        result += data[idx + i * MATRIX_SIZE + 2];
        result += data[idx + i * MATRIX_SIZE + 3];
    }
    data[idx] = result;
}

// Kernel to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += __sinf(data[idx] * 3.14159f);
        result += __cosf(data[idx] * 3.14159f);
        result += __expf(data[idx]);
        result += __logf(data[idx] + 1.0f);
    }
    data[idx] = result;
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[idx], 1.0f);
        atomicAdd(&data[idx], 2.0f);
        atomicAdd(&data[idx], 3.0f);
        atomicAdd(&data[idx], 4.0f);
    }
    data[idx] = result;
}

int main() {
    // Initialize CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);

    // Allocate memory
    float *h_data, *d_data;
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    h_data = (float *)malloc(size);
    cudaMalloc((void **)&d_data, size);

    // Initialize data
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_data[i] = (float)i;
    }
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernels
    int num_blocks = (MATRIX_SIZE * MATRIX_SIZE + 255) / 256;
    dim3 block(256);
    dim3 grid(num_blocks);
    clock_t start_time = clock();
    while ((clock() - start_time) / (double)CLOCKS_PER_SEC < TEST_DURATION) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            stress_compute<<<grid, block, 0, streams[i]>>>(d_data);
            stress_memory<<<grid, block, 0, streams[i]>>>(d_data);
            stress_xu<<<grid, block, 0, streams[i]>>>(d_data);
            stress_atomic<<<grid, block, 0, streams[i]>>>(d_data);
        }
    }

    // Clean up
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);
    free(h_data);

    return 0;
}