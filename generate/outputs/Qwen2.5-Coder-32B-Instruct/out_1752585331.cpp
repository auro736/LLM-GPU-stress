#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

#define BLOCK_SIZE 256
#define MATRIX_SIZE 4096
#define ITERATIONS 10000

__global__ void stressTestKernel(float* A, float* B, float* C, float* D, int matrixSize) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (matrixSize + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (row < matrixSize && m * BLOCK_SIZE + tx < matrixSize)
            s_A[ty][tx] = A[row * matrixSize + m * BLOCK_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < matrixSize && m * BLOCK_SIZE + ty < matrixSize)
            s_B[ty][tx] = B[(m * BLOCK_SIZE + ty) * matrixSize + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < matrixSize && col < matrixSize)
        C[row * matrixSize + col] = Cvalue;

    // Additional computations to maximize resource utilization
    float Dvalue = sinf(Cvalue) + cosf(Cvalue) + tanf(Cvalue);
    atomicAdd(&D[0], Dvalue);
}

int main(int argc, char** argv) {
    int matrixSize = MATRIX_SIZE;
    int iterations = ITERATIONS;

    if (argc > 1)
        matrixSize = std::stoi(argv[1]);
    if (argc > 2)
        iterations = std::stoi(argv[2]);

    size_t bytes = matrixSize * matrixSize * sizeof(float);

    float* h_A = new float[matrixSize * matrixSize];
    float* h_B = new float[matrixSize * matrixSize];
    float* h_C = new float[matrixSize * matrixSize];
    float* h_D = new float[1];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < matrixSize * matrixSize; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    float* d_A, * d_B, * d_C, * d_D;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);
    cudaMalloc((void**)&d_D, sizeof(float));

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);
    cudaMemset(d_D, 0, sizeof(float));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((matrixSize + dimBlock.x - 1) / dimBlock.x, (matrixSize + dimBlock.y - 1) / dimBlock.y);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        stressTestKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_D, matrixSize);
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Stress test completed in " << duration << " ms" << std::endl;
    std::cout << "Final accumulated D value: " << h_D[0] << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}