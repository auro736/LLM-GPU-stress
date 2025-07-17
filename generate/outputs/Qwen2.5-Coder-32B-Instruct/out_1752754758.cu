#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#define BLOCK_SIZE 32

__global__ void stressTestKernel(float* d_matrixA, float* d_matrixB, float* d_matrixC, int N, int iterations) {
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (N - 1) / BLOCK_SIZE + 1; ++m) {
        if (row < N && m * BLOCK_SIZE + tx < N) {
            sharedA[ty][tx] = d_matrixA[row * N + m * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (col < N && m * BLOCK_SIZE + ty < N) {
            sharedB[ty][tx] = d_matrixB[(m * BLOCK_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        atomicAdd(&d_matrixC[row * N + col], Cvalue);
        float specialValue = sinf(d_matrixA[row * N + col]) + cosf(d_matrixB[row * N + col]);
        atomicAdd(&d_matrixC[row * N + col], specialValue);
    }
}

void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void checkCUDAError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <iterations> <duration_in_seconds>" << std::endl;
        return EXIT_FAILURE;
    }

    int N = std::stoi(argv[1]);
    int iterations = std::stoi(argv[2]);
    int duration = std::stoi(argv[3]);

    float* h_matrixA = new float[N * N];
    float* h_matrixB = new float[N * N];
    float* h_matrixC = new float[N * N];

    initializeMatrix(h_matrixA, N);
    initializeMatrix(h_matrixB, N);
    std::fill(h_matrixC, h_matrixC + N * N, 0.0f);

    float* d_matrixA;
    float* d_matrixB;
    float* d_matrixC;

    checkCUDAError(cudaMalloc((void**)&d_matrixA, N * N * sizeof(float)), "Failed to allocate d_matrixA");
    checkCUDAError(cudaMalloc((void**)&d_matrixB, N * N * sizeof(float)), "Failed to allocate d_matrixB");
    checkCUDAError(cudaMalloc((void**)&d_matrixC, N * N * sizeof(float)), "Failed to allocate d_matrixC");

    checkCUDAError(cudaMemcpy(d_matrixA, h_matrixA, N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy d_matrixA");
    checkCUDAError(cudaMemcpy(d_matrixB, h_matrixB, N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy d_matrixB");
    checkCUDAError(cudaMemcpy(d_matrixC, h_matrixC, N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy d_matrixC");

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;

    while (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < duration) {
        for (int i = 0; i < iterations; ++i) {
            stressTestKernel<<<numBlocks, threadsPerBlock>>>(d_matrixA, d_matrixB, d_matrixC, N, iterations);
            cudaDeviceSynchronize();
            checkCUDAError(cudaGetLastError(), "Kernel launch failed");
        }
        end = std::chrono::high_resolution_clock::now();
    }

    checkCUDAError(cudaMemcpy(h_matrixC, d_matrixC, N * N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy d_matrixC");

    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_matrixC;

    checkCUDAError(cudaFree(d_matrixA), "Failed to free d_matrixA");
    checkCUDAError(cudaFree(d_matrixB), "Failed to free d_matrixB");
    checkCUDAError(cudaFree(d_matrixC), "Failed to free d_matrixC");

    return EXIT_SUCCESS;
}