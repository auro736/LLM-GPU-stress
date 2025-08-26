// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

#define MATRIX_SIZE 1024
#define NUM_STREAMS 16
#define NUM_BLOCKS 256
#define NUM_THREADS 256
#define TEST_DURATION 120 // seconds

__global__ void matrixMultiplicationKernel(float *A, float *B, float *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && idy < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += A[idy * size + i] * B[i * size + idx];
        }
        C[idy * size + idx] = sum;
    }
}

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

__global__ void atomicOperationKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        for (int i = 0; i < 1000; i++) {
            atomicAdd(array, 1.0f);
        }
    }
}

int main() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    if (numGPUs < 2) {
        std::cerr << "At least two GPUs are required for this test." << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 9) {
        std::cerr << "At least Ampere or later GPU architecture is required for this test." << std::endl;
        return 1;
    }

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (freeMem < 40 * 1024 * 1024 * 1024) {
        std::cerr << "At least 40 GB of free GPU memory is required for this test." << std::endl;
        return 1;
    }

    float *h_A, *h_B, *h_C, *h_array;
    float *d_A, *d_B, *d_C, *d_array;

    size_t matrixSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    size_t arraySize = MATRIX_SIZE * sizeof(float);

    cudaHostAlloc((void **)&h_A, matrixSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, matrixSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, matrixSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_array, arraySize, cudaHostAllocDefault);

    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);
    cudaMalloc((void **)&d_array, arraySize);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            h_A[i * MATRIX_SIZE + j] = (float)rand() / RAND_MAX;
            h_B[i * MATRIX_SIZE + j] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        h_array[i] = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, arraySize, cudaMemcpyHostToDevice);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < TEST_DURATION) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            int gpu = i % 2;
            cudaSetDevice(gpu);

            matrixMultiplicationKernel<<<NUM_BLOCKS, NUM_THREADS, 0, streams[i]>>>(d_A, d_B, d_C, MATRIX_SIZE);
            floatingPointKernel<<<NUM_BLOCKS, NUM_THREADS, 0, streams[i]>>>(d_array, MATRIX_SIZE);
            atomicOperationKernel<<<NUM_BLOCKS, NUM_THREADS, 0, streams[i]>>>(d_array, MATRIX_SIZE);
        }
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

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