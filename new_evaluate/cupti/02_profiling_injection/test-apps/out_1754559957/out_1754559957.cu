// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

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

__global__ void floatingPointCalculationKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = sqrtf(value) * cosf(value) * sinf(value);
        }
        array[idx] = value;
    }
}

__global__ void atomicOperationKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = 1.0f;
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&array[idx], value);
        }
    }
}

__global__ void specialFunctionKernel(float *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MATRIX_SIZE * MATRIX_SIZE) {
        float value = array[idx];
        for (int i = 0; i < 1000; i++) {
            value = __sinf(value) * __cosf(value) * __expf(value);
        }
        array[idx] = value;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_duration_in_seconds>" << std::endl;
        return 1;
    }

    int testDuration = std::stoi(argv[1]);

    float *h_A, *h_B, *h_C, *h_array;
    float *d_A, *d_B, *d_C, *d_array;

    size_t matrixSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    size_t arraySize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    cudaMallocHost((void **)&h_A, matrixSize);
    cudaMallocHost((void **)&h_B, matrixSize);
    cudaMallocHost((void **)&h_C, matrixSize);
    cudaMallocHost((void **)&h_array, arraySize);

    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);
    cudaMalloc((void **)&d_array, arraySize);

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_array[i] = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array, h_array, arraySize, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    while (true) {
        matrixMultiplicationKernel<<<dim3(GRID_SIZE, GRID_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_A, d_B, d_C);
        floatingPointCalculationKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);
        atomicOperationKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);
        specialFunctionKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_array);

        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (milliseconds / 1000.0f > testDuration) {
            break;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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