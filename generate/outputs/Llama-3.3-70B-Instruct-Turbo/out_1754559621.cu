// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <ctime>

// Define constants
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int MATRIX_SIZE = 1024;
const int NUM_STREAMS = 8;

// Define kernel function for matrix multiplication
__global__ void matrixMultKernel(float* A, float* B, float* C, int size) {
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

// Define kernel function for floating-point calculations
__global__ void floatCalcKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < 1000; i++) {
            val = val * 2.0f - 1.0f;
        }
        data[idx] = val;
    }
}

// Define kernel function for special functions (XU units)
__global__ void specialFuncKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < 1000; i++) {
            val = __sinf(val);
        }
        data[idx] = val;
    }
}

// Define kernel function for atomic operations
__global__ void atomicOpKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&data[idx], 1.0f);
        }
    }
}

int main(int argc, char** argv) {
    // Check if test duration is provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_duration_in_seconds>" << std::endl;
        return 1;
    }

    int testDuration = std::stoi(argv[1]);

    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    // Set CUDA device
    cudaSetDevice(0);

    // Allocate host memory
    float* h_A, *h_B, *h_C, *h_data;
    cudaMallocHost((void**)&h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void**)&h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void**)&h_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void**)&h_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = i;
        h_B[i] = i;
        h_data[i] = i;
    }

    // Allocate device memory
    float* d_A, *d_B, *d_C, *d_data;
    cudaMalloc((void**)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&d_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernels
    clock_t start = clock();
    while ((clock() - start) / (double)CLOCKS_PER_SEC < testDuration) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            matrixMultKernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams[i]>>>(d_A, d_B, d_C, MATRIX_SIZE);
            floatCalcKernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams[i]>>>(d_data, MATRIX_SIZE * MATRIX_SIZE);
            specialFuncKernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams[i]>>>(d_data, MATRIX_SIZE * MATRIX_SIZE);
            atomicOpKernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams[i]>>>(d_data, MATRIX_SIZE * MATRIX_SIZE);
        }
    }

    // Destroy CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_data);

    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_data);

    return 0;
}