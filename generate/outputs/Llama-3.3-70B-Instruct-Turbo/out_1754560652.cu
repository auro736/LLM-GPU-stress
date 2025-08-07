// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_ITERATIONS = 1000;

// Define kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && idy < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += A[idy * size + i] * B[i * size + idx];
        }
        C[idy * size + idx] = sum;
    }
}

// Define kernel for floating-point calculations
__global__ void floatingPointKernel(float* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            result += sinf(array[idx]) * cosf(array[idx]);
        }
        array[idx] = result;
    }
}

// Define kernel for atomic operations
__global__ void atomicKernel(float* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

// Define kernel for L2 cache stress
__global__ void l2CacheKernel(float* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            result += array[(idx + i) % size];
        }
        array[idx] = result;
    }
}

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_duration_in_seconds>" << std::endl;
        return 1;
    }

    int testDuration = std::stoi(argv[1]);

    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA device found" << std::endl;
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using device: " << deviceProp.name << std::endl;

    // Allocate memory on device
    float* d_A, *d_B, *d_C, *d_array;
    cudaMalloc((void**)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_array, MATRIX_SIZE * sizeof(float));

    // Initialize memory on device
    float* h_A = new float[MATRIX_SIZE * MATRIX_SIZE];
    float* h_B = new float[MATRIX_SIZE * MATRIX_SIZE];
    float* h_array = new float[MATRIX_SIZE];
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < MATRIX_SIZE; i++) {
        h_array[i] = rand() / (float)RAND_MAX;
    }
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        matrixMultiplyKernel<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_A, d_B, d_C, MATRIX_SIZE);
        floatingPointKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array, MATRIX_SIZE);
        atomicKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array, MATRIX_SIZE);
        l2CacheKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array, MATRIX_SIZE);

        // Check test duration
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (duration > testDuration) {
            break;
        }
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_array);
    delete[] h_A;
    delete[] h_B;
    delete[] h_array;

    return 0;
}