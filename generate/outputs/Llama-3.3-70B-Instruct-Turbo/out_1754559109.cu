// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_THREADS = BLOCK_SIZE * GRID_SIZE;
const int MEM_SIZE = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

// Define kernel function to stress computational units
__global__ void compute_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += sinf(powf(idx, 2.0f));
        }
        data[idx] = result;
    }
}

// Define kernel function to stress memory
__global__ void memory_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += data[idx];
            data[idx] = result;
        }
    }
}

// Define kernel function to stress XU units
__global__ void xu_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            result += expf(powf(idx, 2.0f));
        }
        data[idx] = result;
    }
}

// Define kernel function to stress atomic operations
__global__ void atomic_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&data[0], 1.0f);
            result += data[0];
        }
        data[idx] = result;
    }
}

// Define kernel function to stress matrix multiplication
__global__ void matrix_stress(float *A, float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_THREADS) {
        float result = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                result += A[i * MATRIX_SIZE + j] * B[j * MATRIX_SIZE + idx % MATRIX_SIZE];
            }
        }
        C[idx] = result;
    }
}

int main(int argc, char **argv) {
    // Check command line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_duration_in_seconds>" << std::endl;
        return 1;
    }

    int test_duration = std::stoi(argv[1]);

    // Initialize CUDA device
    int device_id = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Using device: " << device_prop.name << std::endl;

    // Allocate memory on the device
    float *d_data, *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_data, MEM_SIZE);
    cudaMalloc((void **)&d_A, MEM_SIZE);
    cudaMalloc((void **)&d_B, MEM_SIZE);
    cudaMalloc((void **)&d_C, MEM_SIZE);

    // Launch kernel functions
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        compute_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        memory_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        xu_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        atomic_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        matrix_stress<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C);

        // Check test duration
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > test_duration) {
            break;
        }
    }

    // Free memory on the device
    cudaFree(d_data);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}