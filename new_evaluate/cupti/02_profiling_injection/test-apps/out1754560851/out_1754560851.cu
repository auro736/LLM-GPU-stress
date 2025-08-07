// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <thread>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_ITERATIONS = 1000;

// Define kernel to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result += sinf(data[idx]) * cosf(data[idx]);
    }
    data[idx] = result;
}

// Define kernel to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result += data[idx] * data[idx];
        data[idx] = result;
    }
}

// Define kernel to stress schedulers
__global__ void stress_scheduler(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        __syncwarp();
        result += data[idx] * data[idx];
        data[idx] = result;
    }
}

// Define kernel to stress XU units
__global__ void stress_xu(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result += expf(data[idx]) * logf(data[idx]);
        data[idx] = result;
    }
}

// Define kernel to stress atomic operations
__global__ void stress_atomic(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        atomicAdd(&data[idx], 1.0f);
        result += data[idx];
        data[idx] = result;
    }
}

// Define kernel to stress matrix multiplication
__global__ void stress_matrix_multiply(float *A, float *B, float *C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0.0f;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        result += A[row * MATRIX_SIZE + i] * B[i * MATRIX_SIZE + col];
    }
    C[row * MATRIX_SIZE + col] = result;
}

int main(int argc, char **argv) {
    // Check command line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_duration_in_seconds>" << std::endl;
        return 1;
    }

    // Parse test duration
    int test_duration = std::stoi(argv[1]);

    // Allocate host memory
    float *h_data = new float[GRID_SIZE * BLOCK_SIZE];
    float *h_A = new float[MATRIX_SIZE * MATRIX_SIZE];
    float *h_B = new float[MATRIX_SIZE * MATRIX_SIZE];
    float *h_C = new float[MATRIX_SIZE * MATRIX_SIZE];

    // Initialize host memory
    for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
        h_data[i] = 1.0f;
    }
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Allocate device memory
    float *d_data, *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_data, GRID_SIZE * BLOCK_SIZE * sizeof(float));
    cudaMalloc((void **)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, GRID_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() < test_duration) {
        stress_compute<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        stress_memory<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        stress_scheduler<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        stress_xu<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        stress_atomic<<<GRID_SIZE, BLOCK_SIZE>>>(d_data);
        stress_matrix_multiply<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>> (d_A, d_B, d_C);
    }

    // Copy data from device to host
    cudaMemcpy(h_data, d_data, GRID_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_data;
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}