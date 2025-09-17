// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define constants
#define MATRIX_SIZE 1024
#define NUM_THREADS 256
#define NUM_BLOCKS 256
#define TEST_DURATION 60 // seconds

// Define kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float* A, float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            sum += A[row * MATRIX_SIZE + i] * B[i * MATRIX_SIZE + col];
        }
        C[row * MATRIX_SIZE + col] = sum;
    }
}

// Define kernel for floating-point calculations
__global__ void floatingPointKernel(float* array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = sqrtf(value) * sinf(value) * cosf(value);
        }
        array[idx] = value;
    }
}

// Define kernel for special functions stressing XU units
__global__ void specialFunctionsKernel(float* array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = expf(value) * logf(value) * powf(value, 2.0f);
        }
        array[idx] = value;
    }
}

// Define kernel for atomic operations
__global__ void atomicOperationsKernel(float* array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = 0.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

int main() {
    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }

    // Set CUDA device
    cudaSetDevice(0);

    // Allocate host memory
    float* h_A, *h_B, *h_C, *h_array;
    cudaMallocHost((void**)&h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void**)&h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void**)&h_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void**)&h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_array[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float* d_A, *d_B, *d_C, *d_array;
    cudaMalloc((void**)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 dimBlock(NUM_THREADS, NUM_THREADS);
    dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS);
    clock_t start = clock();
    while ((float)(clock() - start) / CLOCKS_PER_SEC < TEST_DURATION) {
        matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
        floatingPointKernel<<<dimGrid, dimBlock>>>(d_array);
        specialFunctionsKernel<<<dimGrid, dimBlock>>>(d_array);
        atomicOperationsKernel<<<dimGrid, dimBlock>>>(d_array);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_array);

    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_array);

    return 0;
}