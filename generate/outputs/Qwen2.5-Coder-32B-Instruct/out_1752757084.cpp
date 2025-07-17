// stress_test.cu

#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 128

__device__ float matrixA[1024][1024];
__device__ float matrixB[1024][1024];
__device__ float matrixC[1024][1024];

__global__ void matrixMultiplyKernel(int size) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float value = 0.0f;

    for (int m = 0; m < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (row < size && m * BLOCK_SIZE + tx < size) {
            s_A[ty][tx] = matrixA[row][m * BLOCK_SIZE + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if (col < size && m * BLOCK_SIZE + ty < size) {
            s_B[ty][tx] = matrixB[m * BLOCK_SIZE + ty][col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        //__syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += s_A[ty][k] * s_B[k][tx];
        }

        //__syncthreads();
    }

    if (row < size && col < size) {
        atomicAdd(&matrixC[row][col], value);
    }
}

__global__ void specialFunctionsKernel(int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        int row = idx / size;
        int col = idx % size;
        float val = matrixC[row][col];
        matrixC[row][col] = expf(tanf(logf(sqrtf(val * val + 1.0f))));
    }
}

__global__ void initializeMatrixKernel(float *matrix, int size, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        int row = idx / size;
        int col = idx % size;
        curandState localState = state[idx];
        matrix[row][col] = curand_uniform(&localState);
        state[idx] = localState;
    }
}

int main(int argc, char **argv) {
    int size = 1024;
    int duration = 10; // seconds

    if (argc > 1) {
        size = std::stoi(argv[1]);
    }
    if (argc > 2) {
        duration = std::stoi(argv[2]);
    }

    size_t matrixSize = size * size * sizeof(float);
    size_t stateSize = size * size * sizeof(curandState);

    float *h_matrixA, *h_matrixB, *h_matrixC;
    float *d_matrixA, *d_matrixB, *d_matrixC;
    curandState *d_state;

    h_matrixA = (float *)malloc(matrixSize);
    h_matrixB = (float *)malloc(matrixSize);
    h_matrixC = (float *)malloc(matrixSize);

    cudaMalloc((void **)&d_matrixA, matrixSize);
    cudaMalloc((void **)&d_matrixB, matrixSize);
    cudaMalloc((void **)&d_matrixC, matrixSize);
    cudaMalloc((void **)&d_state, stateSize);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    initializeMatrixKernel<<<grid, block>>>(d_matrixA, size, d_state);
    cudaDeviceSynchronize();

    initializeMatrixKernel<<<grid, block>>>(d_matrixB, size, d_state);
    cudaDeviceSynchronize();

    cudaMemset(d_matrixC, 0, matrixSize);

    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        matrixMultiplyKernel<<<grid, block>>>(size);
        specialFunctionsKernel<<<(size * size + 255) / 256, 256>>>(size);

        //cudaDeviceSynchronize();

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

        if (elapsed >= duration) {
            break;
        }
    }

    cudaMemcpy(h_matrixC, d_matrixC, matrixSize, cudaMemcpyDeviceToHost);

    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);
    cudaFree(d_state);
    free(h_matrixA);
    free(h_matrixB);
    free(h_matrixC);

    return 0;
}
```

This CUDA program is designed to stress test modern NVIDIA GPUs by performing intensive matrix multiplications and special function computations. It utilizes shared memory for efficient data handling, ensures memory coalescing, and maximizes occupancy by using a block size of 128x128. The program is configurable for matrix size and test duration, making it suitable for different testing scenarios.