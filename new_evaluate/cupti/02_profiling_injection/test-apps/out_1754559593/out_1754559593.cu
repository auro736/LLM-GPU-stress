// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 16;
const int GRID_SIZE = 256;
const int MATRIX_SIZE = 1024;
const int NUM_ITERATIONS = 1000;

// Kernel to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result += sinf(data[idx]) * cosf(data[idx]);
    }
    data[idx] = result;
}

// Kernel to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result += data[idx * MATRIX_SIZE + threadIdx.x];
    }
    data[idx] = result;
}

// Kernel to stress schedulers
__global__ void stress_scheduler(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        __syncwarp();
        result += data[idx];
    }
    data[idx] = result;
}

// Kernel to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result += __expf(data[idx]) * __logf(data[idx]);
    }
    data[idx] = result;
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        atomicAdd(&data[0], 1.0f);
    }
    data[idx] = result;
}

int main(int argc, char **argv) {
    // Check if test duration is provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_duration_in_seconds>" << std::endl;
        return 1;
    }

    int test_duration = std::stoi(argv[1]);

    // Initialize CUDA
    cudaDeviceReset();
    int device_id = 0;
    cudaSetDevice(device_id);

    // Allocate memory
    float *data;
    cudaMalloc((void **)&data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        stress_compute<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_memory<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_scheduler<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_xu<<<GRID_SIZE, BLOCK_SIZE>>>(data);
        stress_atomic<<<GRID_SIZE, BLOCK_SIZE>>>(data);

        // Check if test duration has been reached
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (elapsed_time >= test_duration) {
            break;
        }
    }

    // Free memory
    cudaFree(data);

    return 0;
}