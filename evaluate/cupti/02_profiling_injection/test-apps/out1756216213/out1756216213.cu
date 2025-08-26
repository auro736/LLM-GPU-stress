// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

#define MATRIX_SIZE 1024
#define NUM_BLOCKS 16
#define NUM_THREADS 256
#define TEST_DURATION 120 // seconds

// Kernel to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx] * data[idx];
        result /= data[idx];
    }
    data[idx] = result;
}

// Kernel to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += data[idx + i];
    }
    data[idx] = result;
}

// Kernel to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += __sinf(data[idx]);
        result += __cosf(data[idx]);
    }
    data[idx] = result;
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&data[idx], 1.0f);
    }
    data[idx] = result;
}

int main() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 2) {
        std::cerr << "At least two GPUs are required." << std::endl;
        return 1;
    }

    float *h_data, *d_data[2];
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    h_data = (float *)malloc(size);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_data[i] = (float)i;
    }

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&d_data[i], size);
        cudaMemcpy(d_data[i], h_data, size, cudaMemcpyHostToDevice);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(i);
            stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(d_data[i]);
            stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(d_data[i]);
            stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(d_data[i]);
            stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_data[i]);
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (duration > TEST_DURATION) {
            break;
        }
    }

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaFree(d_data[i]);
    }
    free(h_data);

    return 0;
}