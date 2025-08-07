// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <iostream>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_ITERATIONS = 1000;

// Define kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}

// Define kernel for floating-point calculations
__global__ void floatingPointKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = sqrtf(value) * sinf(value) * cosf(value);
        }
        array[idx] = value;
    }
}

// Define kernel for special functions stressing XU units
__global__ void specialFunctionsKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = expf(value) * logf(value) * powf(value, 2.0f);
        }
        array[idx] = value;
    }
}

// Define kernel for atomic operations
__global__ void atomicOperationsKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

int main(int argc, char **argv) {
    // Initialize CUDA devices
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    for (int i = 0; i < numDevices; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }

    // Set test duration
    int testDuration;
    if (argc > 1) {
        testDuration = atoi(argv[1]);
    } else {
        testDuration = 60; // default test duration in seconds
    }

    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_array;
    cudaMallocHost((void **)&h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_array[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_array;
    cudaMalloc((void **)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (1) {
        // Launch matrix multiplication kernel
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(GRID_SIZE, GRID_SIZE);
        matrixMultiplyKernel<<<grid, block>>>(d_A, d_B, d_C, MATRIX_SIZE);

        // Launch floating-point calculations kernel
        block = dim3(BLOCK_SIZE);
        grid = dim3(GRID_SIZE);
        floatingPointKernel<<<grid, block>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);

        // Launch special functions kernel
        floatingPointKernel<<<grid, block>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);

        // Launch atomic operations kernel
        atomicOperationsKernel<<<grid, block>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);

        // Check test duration
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        if (elapsed > testDuration) {
            break;
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_array);

    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_array);

    return 0;
}