// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

#define MATRIX_SIZE 1024
#define NUM_ITERATIONS 1000
#define TEST_DURATION 120 // seconds

// Kernel to stress computational units
__global__ void compute_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            result += sinf(data[idx]) * cosf(data[idx]);
        }
        data[idx] = result;
    }
}

// Kernel to stress memory
__global__ void memory_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            result += data[idx] * data[idx];
            data[idx] = result;
        }
    }
}

// Kernel to stress schedulers
__global__ void scheduler_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            __syncwarp();
            result += data[idx] * data[idx];
            data[idx] = result;
        }
    }
}

// Kernel to stress XU units
__global__ void xu_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            result += expf(data[idx]) * logf(data[idx]);
            data[idx] = result;
        }
    }
}

// Kernel to stress atomic operations
__global__ void atomic_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            atomicAdd(&data[idx], 1.0f);
            result += data[idx];
            data[idx] = result;
        }
    }
}

int main() {
    // Initialize CUDA devices
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }

    // Allocate memory on each device
    float *data[2];
    cudaSetDevice(0);
    cudaMalloc((void **)&data[0], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaSetDevice(1);
    cudaMalloc((void **)&data[1], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize data
    cudaSetDevice(0);
    cudaMemset(data[0], 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaSetDevice(1);
    cudaMemset(data[1], 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }

        cudaSetDevice(0);
        compute_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[0]);
        memory_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[0]);
        scheduler_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[0]);
        xu_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[0]);
        atomic_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[0]);

        cudaSetDevice(1);
        compute_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[1]);
        memory_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[1]);
        scheduler_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[1]);
        xu_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[1]);
        atomic_stress<<<MATRIX_SIZE, MATRIX_SIZE>>>(data[1]);
    }

    // Clean up
    cudaSetDevice(0);
    cudaFree(data[0]);
    cudaSetDevice(1);
    cudaFree(data[1]);

    return 0;
}