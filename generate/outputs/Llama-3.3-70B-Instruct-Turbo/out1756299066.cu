// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

// Define constants
const int NUM_GPUS = 2;
const int MATRIX_SIZE = 1024;
const int NUM_ITERATIONS = 1000;
const int TEST_DURATION = 120; // seconds

// Define kernel for matrix multiplication
__global__ void matrixMultKernel(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}

// Define kernel for floating-point calculations
__global__ void floatCalcKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sqrtf(data[idx] * data[idx]);
        }
        data[idx] = result;
    }
}

// Define kernel for special functions (XU units)
__global__ void specialFuncKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(data[idx] * data[idx]);
        }
        data[idx] = result;
    }
}

// Define kernel for atomic operations
__global__ void atomicOpKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&data[idx], 1.0f);
        }
    }
}

int main() {
    // Initialize CUDA devices
    cudaDeviceProp prop;
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
    }

    // Allocate memory on each GPU
    float *A[NUM_GPUS], *B[NUM_GPUS], *C[NUM_GPUS], *data[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        cudaMalloc((void **)&B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        cudaMalloc((void **)&C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        cudaMalloc((void **)&data[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    }

    // Initialize data
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemset(A[i], 1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        cudaMemset(B[i], 2, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        cudaMemset(data[i], 3.0f, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    }

    // Start test
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        // Check if test duration has been reached
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }

        // Launch kernels on each GPU
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            dim3 block(16, 16);
            dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);
            matrixMultKernel<<<grid, block>>>(A[i], B[i], C[i], MATRIX_SIZE);
            floatCalcKernel<<<grid, block>>>(data[i], MATRIX_SIZE * MATRIX_SIZE);
            specialFuncKernel<<<grid, block>>>(data[i], MATRIX_SIZE * MATRIX_SIZE);
            atomicOpKernel<<<grid, block>>>(data[i], MATRIX_SIZE * MATRIX_SIZE);
        }
    }

    // Clean up
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaFree(A[i]);
        cudaFree(B[i]);
        cudaFree(C[i]);
        cudaFree(data[i]);
    }

    return 0;
}