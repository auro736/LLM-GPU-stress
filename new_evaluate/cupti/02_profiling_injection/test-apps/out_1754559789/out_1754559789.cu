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

// Define kernel for atomic operations
__global__ void atomicKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

// Define kernel for L2 cache stress
__global__ void l2CacheKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = array[(idx + i) % size];
        }
        array[idx] = value;
    }
}

int main(int argc, char **argv) {
    // Check command line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_duration_in_seconds>" << std::endl;
        return 1;
    }

    int testDuration = std::stoi(argv[1]);

    // Initialize CUDA devices
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    for (int i = 0; i < numDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }

    // Set device
    cudaSetDevice(0);

    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_array;
    cudaMallocHost((void **)&h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = i;
        h_B[i] = i;
        h_C[i] = 0.0f;
        h_array[i] = i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_array;
    cudaMalloc((void **)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        matrixMultiplyKernel<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_A, d_B, d_C, MATRIX_SIZE);
        floatingPointKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);
        atomicKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);
        l2CacheKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array, MATRIX_SIZE * MATRIX_SIZE);

        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (duration > testDuration) {
            break;
        }
    }

    // Copy data from device to host
    cudaMemcpy(h_C, d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_array, d_array, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_array);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_array);

    return 0;
}