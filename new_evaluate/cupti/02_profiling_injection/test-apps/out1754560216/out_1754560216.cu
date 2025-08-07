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

__global__ void matrixMultiplicationKernel(float *A, float *B, float *C) {
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

__global__ void floatingPointCalculationKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = value * 2.0f - 1.0f;
        }
        array[idx] = value;
    }
}

__global__ void specialFunctionKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = __sinf(value);
        }
        array[idx] = value;
    }
}

__global__ void atomicOperationKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

int main(int argc, char **argv) {
    int duration = TEST_DURATION;
    if (argc > 1) {
        duration = std::stoi(argv[1]);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_array;
    cudaMalloc((void **)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize device memory
    float *h_A, *h_B, *h_array;
    h_A = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    h_B = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    h_array = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_array[i] = 3.0f;
    }
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        matrixMultiplicationKernel<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_A, d_B, d_C);
        floatingPointCalculationKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);
        specialFunctionKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);
        atomicOperationKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);

        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (elapsed_time > duration) {
            break;
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_array);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_array);

    return 0;
}