// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int MATRIX_SIZE = 1024;
const int NUM_THREADS = 256;
const int NUM_BLOCKS = 256;
const int TEST_DURATION = 120; // seconds

// Kernel function to stress computational units
__global__ void stress_compute(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            result += matrix[idx * MATRIX_SIZE + i] * matrix[i * MATRIX_SIZE + idy];
        }
        matrix[idx * MATRIX_SIZE + idy] = result;
    }
}

// Kernel function to stress memory
__global__ void stress_memory(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            result += matrix[idx * MATRIX_SIZE + i] * matrix[i * MATRIX_SIZE + idy];
        }
        __shared__ float shared_result;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            shared_result = result;
        }
        __syncthreads();
        matrix[idx * MATRIX_SIZE + idy] = shared_result;
    }
}

// Kernel function to stress schedulers
__global__ void stress_scheduler(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            result += matrix[idx * MATRIX_SIZE + i] * matrix[i * MATRIX_SIZE + idy];
        }
        atomicAdd(&matrix[idx * MATRIX_SIZE + idy], result);
    }
}

int main() {
    // Initialize CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        std::cerr << "At least two GPUs are required for this test." << std::endl;
        return 1;
    }

    // Allocate memory on each device
    float *matrix0, *matrix1;
    cudaSetDevice(0);
    cudaMalloc((void **)&matrix0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaSetDevice(1);
    cudaMalloc((void **)&matrix1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize matrices
    cudaSetDevice(0);
    cudaMemset(matrix0, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaSetDevice(1);
    cudaMemset(matrix1, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Start test
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        // Launch kernels on each device
        cudaSetDevice(0);
        stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(matrix0);
        stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(matrix0);
        stress_scheduler<<<NUM_BLOCKS, NUM_THREADS>>>(matrix0);

        cudaSetDevice(1);
        stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(matrix1);
        stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(matrix1);
        stress_scheduler<<<NUM_BLOCKS, NUM_THREADS>>>(matrix1);
    }

    // Clean up
    cudaSetDevice(0);
    cudaFree(matrix0);
    cudaSetDevice(1);
    cudaFree(matrix1);

    return 0;
}