// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define matrix dimensions
#define MATRIX_DIM 1024

// Define block dimensions
#define BLOCK_DIM 16

// Define number of blocks
#define NUM_BLOCKS 256

// Define number of iterations
#define NUM_ITERATIONS 1000

// Define test duration time in seconds
#define TEST_DURATION 60

// Kernel function to stress computational units
__global__ void stress_compute(float *matrix1, float *matrix2, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_DIM && idy < MATRIX_DIM) {
        float sum = 0.0f;
        for (int i = 0; i < MATRIX_DIM; i++) {
            sum += matrix1[idy * MATRIX_DIM + i] * matrix2[i * MATRIX_DIM + idx];
        }
        result[idy * MATRIX_DIM + idx] = sum;
    }
}

// Kernel function to stress memory
__global__ void stress_memory(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_DIM * MATRIX_DIM) {
        array[idx] = array[idx] * 2.0f;
    }
}

// Kernel function to stress XU units
__global__ void stress_xu(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_DIM * MATRIX_DIM) {
        array[idx] = sinf(array[idx]);
    }
}

// Kernel function to stress atomic operations
__global__ void stress_atomic(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MATRIX_DIM * MATRIX_DIM) {
        atomicAdd(&array[idx], 1.0f);
    }
}

int main() {
    // Allocate host memory
    float *h_matrix1, *h_matrix2, *h_result;
    h_matrix1 = new float[MATRIX_DIM * MATRIX_DIM];
    h_matrix2 = new float[MATRIX_DIM * MATRIX_DIM];
    h_result = new float[MATRIX_DIM * MATRIX_DIM];

    // Initialize host memory
    for (int i = 0; i < MATRIX_DIM * MATRIX_DIM; i++) {
        h_matrix1[i] = 1.0f;
        h_matrix2[i] = 2.0f;
        h_result[i] = 0.0f;
    }

    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_result, *d_array;
    cudaMalloc((void **)&d_matrix1, MATRIX_DIM * MATRIX_DIM * sizeof(float));
    cudaMalloc((void **)&d_matrix2, MATRIX_DIM * MATRIX_DIM * sizeof(float));
    cudaMalloc((void **)&d_result, MATRIX_DIM * MATRIX_DIM * sizeof(float));
    cudaMalloc((void **)&d_array, MATRIX_DIM * MATRIX_DIM * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_matrix1, h_matrix1, MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2, MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_matrix1, MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        stress_compute<<<dim3(NUM_BLOCKS, NUM_BLOCKS), dim3(BLOCK_DIM, BLOCK_DIM)>>>(d_matrix1, d_matrix2, d_result);
        stress_memory<<<NUM_BLOCKS, BLOCK_DIM>>>(d_array);
        stress_xu<<<NUM_BLOCKS, BLOCK_DIM>>>(d_array);
        stress_atomic<<<NUM_BLOCKS, BLOCK_DIM>>>(d_array);
    }

    // Copy device memory to host memory
    cudaMemcpy(h_result, d_result, MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Free host memory
    delete[] h_matrix1;
    delete[] h_matrix2;
    delete[] h_result;

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);
    cudaFree(d_array);

    return 0;
}