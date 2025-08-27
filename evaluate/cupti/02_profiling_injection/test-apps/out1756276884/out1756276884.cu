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
        }
        data[idx] = result;
    }
}

// Kernel to stress XU units
__global__ void xu_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            result += __sinf(data[idx]) * __cosf(data[idx]);
        }
        data[idx] = result;
    }
}

// Kernel to stress atomic operations
__global__ void atomic_stress(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            atomicAdd(&data[idx], 1.0f);
        }
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        std::cerr << "At least two GPUs are required" << std::endl;
        return 1;
    }

    cudaDeviceProp deviceProp;
    for (int device = 0; device < deviceCount; device++) {
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
    }

    // Allocate memory on each device
    float *data0, *data1;
    cudaMalloc((void **)&data0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&data1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize data
    cudaMemset(data0, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMemset(data1, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Launch kernels on each device
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        compute_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream0>>>(data0);
        memory_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream0>>>(data0);
        xu_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream0>>>(data0);
        atomic_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream0>>>(data0);

        compute_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream1>>>(data1);
        memory_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream1>>>(data1);
        xu_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream1>>>(data1);
        atomic_stress<<<MATRIX_SIZE, MATRIX_SIZE, 0, stream1>>>(data1);
    }

    // Clean up
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaFree(data0);
    cudaFree(data1);

    return 0;
}