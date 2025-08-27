// gpu_stress_test.cu

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cmath>
#include <ctime>

// Define constants
const int MATRIX_SIZE = 1024;
const int NUM_BLOCKS = 16;
const int NUM_THREADS = 256;
const int TEST_DURATION = 120; // seconds

// Kernel function to stress computational units
__global__ void stress_compute(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            val = val * 2.0f + sinf(val) * cosf(val);
        }
        matrix[idx] = val;
    }
}

// Kernel function to stress memory
__global__ void stress_memory(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            val = val * 2.0f;
            matrix[idx] = val;
            __syncthreads();
        }
    }
}

// Kernel function to stress XU units
__global__ void stress_xu(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            val = sqrtf(val) * logf(val) * expf(val);
        }
        matrix[idx] = val;
    }
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float val = matrix[idx];
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&matrix[idx], val);
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

    // Set test duration
    clock_t start_time = clock();

    // Allocate memory on each device
    float *matrix[2];
    cudaDevice_t devices[2];
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&matrix[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    }

    // Launch kernels on each device
    while ((clock() - start_time) / CLOCKS_PER_SEC < TEST_DURATION) {
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(i);
            stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
            stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
            stress_xu<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
            stress_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(matrix[i]);
        }
    }

    // Free memory on each device
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaFree(matrix[i]);
    }

    return 0;
}