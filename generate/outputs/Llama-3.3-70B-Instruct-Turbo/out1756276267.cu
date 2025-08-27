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

// Kernel function to stress computational units
__global__ void stress_compute(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            sum += matrix[idx * MATRIX_SIZE + idy] * matrix[idy * MATRIX_SIZE + idx];
        }
        matrix[idx * MATRIX_SIZE + idy] = sum;
    }
}

// Kernel function to stress memory
__global__ void stress_memory(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float value = matrix[idx * MATRIX_SIZE + idy];
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            matrix[idx * MATRIX_SIZE + idy] = value * 2.0f;
            value = matrix[idx * MATRIX_SIZE + idy];
        }
    }
}

// Kernel function to stress XU units
__global__ void stress_xu(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float value = matrix[idx * MATRIX_SIZE + idy];
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            value = __sinf(value);
            matrix[idx * MATRIX_SIZE + idy] = value;
        }
    }
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float value = matrix[idx * MATRIX_SIZE + idy];
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            atomicAdd(&matrix[idx * MATRIX_SIZE + idy], value);
        }
    }
}

int main() {
    int num_gpus = 2;
    int device;
    cudaDeviceProp prop;

    // Initialize devices
    for (device = 0; device < num_gpus; device++) {
        cudaSetDevice(device);
        cudaGetDeviceProperties(&prop, device);
        std::cout << "Device " << device << ": " << prop.name << std::endl;
    }

    // Allocate memory
    float *matrix;
    cudaMalloc((void **)&matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize matrix
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i * MATRIX_SIZE + j] = (float)rand() / RAND_MAX;
        }
    }

    // Launch kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (true) {
        for (device = 0; device < num_gpus; device++) {
            cudaSetDevice(device);
            stress_compute<<<dim3(16, 16), dim3(16, 16)>>>(matrix);
            stress_memory<<<dim3(16, 16), dim3(16, 16)>>>(matrix);
            stress_xu<<<dim3(16, 16), dim3(16, 16)>>>(matrix);
            stress_atomic<<<dim3(16, 16), dim3(16, 16)>>>(matrix);
        }

        // Check test duration
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (milliseconds / 1000.0f > TEST_DURATION) {
            break;
        }
    }

    // Clean up
    cudaFree(matrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}