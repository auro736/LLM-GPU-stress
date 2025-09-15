// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <thread>

#define BLOCK_SIZE 256
#define GRID_SIZE 1024
#define MATRIX_SIZE 1024
#define TEST_DURATION 60 // seconds

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            sum += A[idy * MATRIX_SIZE + i] * B[i * MATRIX_SIZE + idx];
        }
        C[idy * MATRIX_SIZE + idx] = sum;
    }
}

__global__ void floatingPointCalculationKernel(float* array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(array[idx]) * cosf(array[idx]);
        }
        array[idx] = result;
    }
}

__global__ void atomicOperationKernel(float* array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = 1.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

int main() {
    // Initialize CUDA
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Device: " << deviceProp.name << std::endl;

    // Allocate memory
    float* A, *B, *C, *array;
    cudaMalloc((void**)&A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        array[i] = 3.0f;
    }

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        matrixMultiplicationKernel<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(A, B, C);
        floatingPointCalculationKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(array);
        atomicOperationKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(array);
    }

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(array);

    return 0;
}