// cuda_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// Define constants
#define BLOCK_SIZE 256
#define GRID_SIZE 256
#define MATRIX_SIZE 1024
#define TEST_DURATION 60 // seconds

// Define kernel for matrix multiplication
__global__ void matrixMultKernel(float *A, float *B, float *C, int size) {
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
__global__ void floatCalcKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        val = sinf(val) + cosf(val) + expf(val) + logf(val);
        data[idx] = val;
    }
}

// Define kernel for special functions stressing XU units
__global__ void specialFuncKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        val = sqrtf(val) + rsqrtf(val) + rcbrtf(val);
        data[idx] = val;
    }
}

// Define kernel for atomic operations
__global__ void atomicOpKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        atomicAdd(&data[idx], val);
    }
}

// Define kernel for memory access pattern
__global__ void memoryAccessKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx + size / 2] = val;
        __syncthreads();
        val = data[idx + size / 2];
        data[idx] = val;
    }
}

int main() {
    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0);

    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_data;
    cudaMallocHost((void **)&h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize host data
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)i;
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_data;
    cudaMalloc((void **)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(GRID_SIZE, GRID_SIZE);
    for (int i = 0; i < TEST_DURATION; i++) {
        matrixMultKernel<<<grid, block>>>(d_A, d_B, d_C, MATRIX_SIZE);
        floatCalcKernel<<<grid, block>>>(d_data, MATRIX_SIZE * MATRIX_SIZE);
        specialFuncKernel<<<grid, block>>>(d_data, MATRIX_SIZE * MATRIX_SIZE);
        atomicOpKernel<<<grid, block>>>(d_data, MATRIX_SIZE * MATRIX_SIZE);
        memoryAccessKernel<<<grid, block>>>(d_data, MATRIX_SIZE * MATRIX_SIZE);
        cudaDeviceSynchronize();
    }

    // Transfer data from device to host
    cudaMemcpy(h_C, d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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