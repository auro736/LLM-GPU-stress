// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_ITERATIONS = 1000;

// Define kernel for matrix multiplication
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

// Define kernel for floating-point calculations
__global__ void floatingPointKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = value * 2.0f - 3.0f * value * value;
        }
        array[idx] = value;
    }
}

// Define kernel for special functions stressing XU units
__global__ void specialFunctionKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = __sinf(value) * __cosf(value);
        }
        array[idx] = value;
    }
}

// Define kernel for atomic operations
__global__ void atomicOperationKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = 0.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

int main(int argc, char **argv) {
    int testDuration = 60; // Default test duration in seconds
    if (argc > 1) {
        testDuration = std::stoi(argv[1]);
    }

    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_array;
    cudaMallocHost((void **)&h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_array[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_array;
    cudaMalloc((void **)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < testDuration) {
        matrixMultiplicationKernel<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_A, d_B, d_C);
        floatingPointKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);
        specialFunctionKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);
        atomicOperationKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);
    }

    // Copy device memory to host memory
    cudaMemcpy(h_C, d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_array, d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free host and device memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_array);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_array);

    return 0;
}