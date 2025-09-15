// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

#define MATRIX_SIZE 1024
#define NUM_THREADS 256
#define NUM_BLOCKS 256
#define TEST_DURATION 60 // seconds

__global__ void matrixMultiplicationKernel(float *A, float *B, float *C) {
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

__global__ void floatingPointKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(array[idx]) * cosf(array[idx]);
        }
        array[idx] = result;
    }
}

__global__ void atomicOperationKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = 1.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

__global__ void memoryAccessKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            array[idx] = value * 2.0f;
            value = array[idx];
        }
    }
}

int main() {
    // Initialize CUDA
    cudaDeviceReset();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // Allocate memory
    float *A, *B, *C, *array;
    cudaMalloc((void **)&A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        array[i] = 3.0f;
    }

    // Launch kernels
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + std::chrono::seconds(TEST_DURATION);
    while (std::chrono::high_resolution_clock::now() < end) {
        matrixMultiplicationKernel<<<dim3(NUM_BLOCKS, NUM_BLOCKS), dim3(NUM_THREADS, NUM_THREADS)>>>(A, B, C);
        floatingPointKernel<<<dim3(NUM_BLOCKS, NUM_BLOCKS), dim3(NUM_THREADS, NUM_THREADS)>>>(array);
        atomicOperationKernel<<<dim3(NUM_BLOCKS, NUM_BLOCKS), dim3(NUM_THREADS, NUM_THREADS)>>>(array);
        memoryAccessKernel<<<dim3(NUM_BLOCKS, NUM_BLOCKS), dim3(NUM_THREADS, NUM_THREADS)>>>(array);
    }

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(array);

    // Cleanup
    cudaDeviceReset();

    return 0;
}