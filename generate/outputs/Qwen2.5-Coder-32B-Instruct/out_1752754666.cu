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

        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += sharedA[ty][e] * sharedB[e][tx];
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

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <iterations> <test_duration>" << std::endl;
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    int testDuration = atoi(argv[3]);

    size_t size = N * N * sizeof(float);

    float* h_matrixA = new float[N * N];
    float* h_matrixB = new float[N * N];
    float* h_matrixC = new float[N * N];

    initializeMatrix(h_matrixA, N);
    initializeMatrix(h_matrixB, N);
    memset(h_matrixC, 0, size);

    float* d_matrixA;
    float* d_matrixB;
    float* d_matrixC;

    cudaMalloc((void**)&d_matrixA, size);
    cudaMalloc((void**)&d_matrixB, size);
    cudaMalloc((void**)&d_matrixC, size);

    cudaMemcpy(d_matrixA, h_matrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixC, h_matrixC, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedTime = 0.0;

    while (elapsedTime < testDuration) {
        for (int i = 0; i < iterations; ++i) {
            stressTestKernel<<<dimGrid, dimBlock>>>(d_matrixA, d_matrixB, d_matrixC, N, iterations);
            checkCUDAError("Kernel launch");
        }

        cudaDeviceSynchronize();
        checkCUDAError("cudaDeviceSynchronize");

        endTime = std::chrono::high_resolution_clock::now();
        elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    }

    cudaMemcpy(h_matrixC, d_matrixC, size, cudaMemcpyDeviceToHost);

    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);

    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_matrixC;

    return EXIT_SUCCESS;
}