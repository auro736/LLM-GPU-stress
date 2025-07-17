#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#define BLOCK_SIZE 32

__device__ float matrixA[1024][1024];
__device__ float matrixB[1024][1024];
__device__ float matrixC[1024][1024];

__global__ void initializeMatrices(int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) {
        curandState state;
        curand_init(clock64(), row * size + col, 0, &state);
        matrixA[row][col] = curand_uniform(&state);
        matrixB[row][col] = curand_uniform(&state);
        matrixC[row][col] = 0.0f;
    }
}

__global__ void matrixMultiply(int size) {
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float value = 0.0f;

    for (int m = 0; m < (size - 1) / BLOCK_SIZE + 1; ++m) {
        if (row < size && m * BLOCK_SIZE + tx < size)
            sharedA[ty][tx] = matrixA[row][m * BLOCK_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;

        if (col < size && m * BLOCK_SIZE + ty < size)
            sharedB[ty][tx] = matrixB[m * BLOCK_SIZE + ty][col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            value += sharedA[ty][k] * sharedB[k][tx];

        __syncthreads();
    }

    if (row < size && col < size)
        matrixC[row][col] = value;
}

__global__ void atomicOperations(int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        atomicAdd(&matrixC[idx / size][idx % size], sinf(matrixA[idx / size][idx % size]) + cosf(matrixB[idx / size][idx % size]));
    }
}

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int matrixSize = 1024;
    int numIterations = 1000;
    if (argc > 1) matrixSize = atoi(argv[1]);
    if (argc > 2) numIterations = atoi(argv[2]);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((matrixSize - 1) / BLOCK_SIZE + 1, (matrixSize - 1) / BLOCK_SIZE + 1);

    initializeMatrices<<<gridSize, blockSize>>>(matrixSize);
    checkCUDAError("initializeMatrices");

    for (int i = 0; i < numIterations; ++i) {
        matrixMultiply<<<gridSize, blockSize>>>(matrixSize);
        checkCUDAError("matrixMultiply");

        atomicOperations<<<(matrixSize * matrixSize - 1) / 256 + 1, 256>>>(matrixSize);
        checkCUDAError("atomicOperations");
    }

    cudaDeviceSynchronize();
    cudaFree(0); // Free all uninitialized memory
    return 0;
}