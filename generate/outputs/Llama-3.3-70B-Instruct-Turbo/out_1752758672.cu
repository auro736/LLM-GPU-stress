// cuda_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// Define configurable parameters
#define TEST_DURATION 60 // seconds
#define WORKLOAD_COMPOSITION 0.5 // matrix multiplication weight (0.0 - 1.0)
#define MATRIX_SIZE 1024
#define NUM_THREADS 256
#define NUM_BLOCKS 256

// Define a function to perform matrix multiplication
__global__ void matrixMultiplication(float *A, float *B, float *C, int size) {
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

// Define a function to perform floating-point calculations
__global__ void floatingPointCalculations(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float value! = array[idx];
        value = sqrtf(value);
        value = sinf(value);
        value = expf(value);
        value = logf(value);
        array[idx] = value;
    }
}

// Define a function to stress the XU units
__global__ void xuStress(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < 100; i++) {
            value = sqrtf(value);
            value = sinf(value);
            value = expf(value);
            value = logf(value);
        }
        array[idx] = value;
    }
}

// Define a function to perform atomic operations
__global__ void atomicOperations(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < 100; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

int main() {
    // Initialize CUDA
    cudaDeviceReset();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

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

    // Launch kernel
    dim3 blockSize(NUM_THREADS, NUM_THREADS);
    dim3 gridsize(NUM_BLOCKS, NUM_BLOCKS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float workloadComposition = WORKLOAD_COMPOSITION;
    int testDuration = TEST_DURATION;

    // Record start time
    cudaEventRecord(start, 0);

    while (true) {
        // Check if test duration has expired
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (milliseconds / 1000.0f > testDuration) {
            break;
        }

        // Launch matrix multiplication kernel
        if (workloadComposition > (float)rand() / RAND_MAX) {
            matrixMultiplication<<<gridsize, blockSize>>>(d_A, d_B, d_C, MATRIX_SIZE);
        }

        // Launch floating-point calculations kernel
        floatingPointCalculations<<<gridsize, blockSize>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);

        // Launch XU stress kernel
        xuStress<<<gridsize, blockSize>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);

        // Launch atomic operations kernel
        atomicOperations<<<gridsize, blockSize>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);
    }

    // Copy device memory back to host memory
    cudaMemcpy(h_C, d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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

    // Cleanup
    cudaDeviceReset();

    return 0;
}