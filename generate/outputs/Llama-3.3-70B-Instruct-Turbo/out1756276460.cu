// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define matrix dimensions
#define MATRIX_DIM 1024

// Define number of iterations
#define NUM_ITERATIONS 1000

// Define test duration time in seconds
#define TEST_DURATION 120

// Kernel to stress computational units
__global__ void stress_compute(float *A, float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_DIM && idy < MATRIX_DIM) {
        float sum = 0.0f;
        for (int i = 0; i < MATRIX_DIM; i++) {
            sum += A[idy * MATRIX_DIM + i] * B[i * MATRIX_DIM + idx];
        }
        C[idy * MATRIX_DIM + idx] = sum;
    }
}

// Kernel to stress memory
__global__ void stress_memory(float *A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_DIM * MATRIX_DIM) {
        A[idx] = A[idx] * 2.0f;
    }
}

// Kernel to stress XU units
__global__ void stress_xu(float *A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_DIM * MATRIX_DIM) {
        A[idx] = __sinf(A[idx]);
    }
}

// Kernel to stress atomic operations
__global__ void stress_atomic(float *A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_DIM * MATRIX_DIM) {
        atomicAdd(&A[idx], 1.0f);
    }
}

int main() {
    // Initialize CUDA devices
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    // Allocate host memory
    float *h_A, *h_B, *h_C;
    cudaMallocHost((void **)&h_A, MATRIX_DIM * MATRIX_DIM * sizeof(float));
    cudaMallocHost((void **)&h_B, MATRIX_DIM * MATRIX_DIM * sizeof(float));
    cudaMallocHost((void **)&h_C, MATRIX_DIM * MATRIX_DIM * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_DIM * MATRIX_DIM; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, MATRIX_DIM * MATRIX_DIM * sizeof(float));
    cudaMalloc((void **)&d_B, MATRIX_DIM * MATRIX_DIM * sizeof(float));
    cudaMalloc((void **)&d_C, MATRIX_DIM * MATRIX_DIM * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Set up kernel launch parameters
    dim3 block_dim(16, 16);
    dim3 grid_dim((MATRIX_DIM + block_dim.x - 1) / block_dim.x, (MATRIX_DIM + block_dim.y - 1) / block_dim.y);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < TEST_DURATION) {
        stress_compute<<<grid_dim, block_dim>>>(d_A, d_B, d_C);
        stress_memory<<<grid_dim, block_dim>>>(d_A);
        stress_xu<<<grid_dim, block_dim>>>(d_A);
        stress_atomic<<<grid_dim, block_dim>>>(d_A);
    }

    // Copy device memory to host memory
    cudaMemcpy(h_C, d_C, MATRIX_DIM * MATRIX_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}