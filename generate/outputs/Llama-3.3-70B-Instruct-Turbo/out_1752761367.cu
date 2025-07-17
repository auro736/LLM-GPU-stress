// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 256
#define MATRIX_SIZE 1024
#define TEST_DURATION 60 // seconds

__global__ void matrixMultiplicationKernel(float *A, float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < MATRIX_SIZE && idy < MATRIX_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            sum += A[idy * MATRIX_SIZE + i] * B[i * MATRIX_SIZE + idx];
        }
        C[idy * MATRIX_SIZE + idx] = sum;
    }
}

__global__ void floatingPointCalculationsKernel(float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = x[idx];
        for (int i = 0; i < 1000; i++) {
            result = sqrtf(result) * sinf(result) * cosf(result);
        }
        x[idx] = result;
    }
}

__global__ void atomicOperationsKernel(float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        for (int i = 0; i < 1000; i++) {
            //__syncthreads();
            atomicAdd(&x[idx], 1.0f);
        }
    }
}

__global__ void memoryAccessKernel(float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float result = x[idx];
        for (int i = 0; i < 1000; i++) {
            result += x[(idx + i) % (MATRIX_SIZE * MATRIX_SIZE)];
        }
        x[idx] = result;
    }
}

int main(int argc, char **argv) {
    int testDuration = TEST_DURATION;
    if (argc > 1) {
        testDuration = atoi(argv[1]);
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    float *A, *B, *C, *x;
    cudaMalloc((void **)&A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&x, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    cudaMemset(A, 1, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMemset(B, 2, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMemset(x, 3, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    clock_t start = clock();

    while ((clock() - start) / (double)CLOCKS_PER_SEC < testDuration) {
        matrixMultiplicationKernel<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(A, B, C);
        floatingPointCalculationsKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(x);
        atomicOperationsKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(x);
        memoryAccessKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(x);
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(x);

    return 0;
}
