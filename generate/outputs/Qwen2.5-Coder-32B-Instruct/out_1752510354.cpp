#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cmath>
#include <chrono>
#include <vector>

#define BLOCK_SIZE 256
#define TILE_SIZE 32

__constant__ float const_matrix_B[TILE_SIZE * TILE_SIZE];

__global__ void matrixMulKernel(float* C, float* A, int width) {
    __shared__ float shared_matrix_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_matrix_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float Csub = 0;
    for (int m = 0; m < (width - 1) / TILE_SIZE + 1; ++m) {
        if (row < width && m * TILE_SIZE + tx < width) {
            shared_matrix_A[ty][tx] = A[row * width + m * TILE_SIZE + tx];
        } else {
            shared_matrix_A[ty][tx] = 0.0f;
        }

        if (col < width && m * TILE_SIZE + ty < width) {
            shared_matrix_B[ty][tx] = const_matrix_B[(m * TILE_SIZE + ty) * TILE_SIZE + tx];
        } else {
            shared_matrix_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int e = 0; e < TILE_SIZE; ++e) {
            Csub += shared_matrix_A[ty][e] * shared_matrix_B[e][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        atomicAdd(&C[row * width + col], Csub);
    }
}

__global__ void randomAccessKernel(float* data, int n, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState localState = states[idx];
        data[idx] = sinf(curand_uniform(&localState)) * tanf(curand_uniform(&localState));
        states[idx] = localState;
    }
}

__global__ void initRandomStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int width = 4096;
    int n = width * width;
    int duration = 60; // seconds
    int seed = 12345;

    if (argc > 1) width = std::stoi(argv[1]);
    if (argc > 2) duration = std::stoi(argv[2]);
    if (argc > 3) seed = std::stoi(argv[3]);

    float *A, *B, *C, *h_A, *h_B, *h_C;
    curandState* d_states;
    size_t size = width * width * sizeof(float);
    size_t state_size = n * sizeof(curandState);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }

    checkCudaError(cudaMalloc((void**)&A, size), "Failed to allocate A");
    checkCudaError(cudaMalloc((void**)&B, size), "Failed to allocate B");
    checkCudaError(cudaMalloc((void**)&C, size), "Failed to allocate C");
    checkCudaError(cudaMalloc((void**)&d_states, state_size), "Failed to allocate random states");

    checkCudaError(cudaMemcpy(A, h_A, size, cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(B, h_B, size, cudaMemcpyHostToDevice), "Failed to copy B to device");
    checkCudaError(cudaMemcpy(C, h_C, size, cudaMemcpyHostToDevice), "Failed to copy C to device");

    initRandomStates<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, seed, n);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);

    checkCudaError(cudaMemcpyToSymbol(const_matrix_B, B, TILE_SIZE * TILE_SIZE * sizeof(float)), "Failed to copy B to constant memory");

    auto start = std::chrono::high_resolution_clock::now();
    float gflops = 0.0f;
    int iter = 0;

    while (true) {
        matrixMulKernel<<<dimGrid, dimBlock>>>(C, A, width);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");

        randomAccessKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(C, n, d_states);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");

        iter++;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (elapsed.count() >= duration) {
            break;
        }

        gflops = (2.0 * static_cast<float>(width) * static_cast<float>(width) * static_cast<float>(width) * iter) / (elapsed.count() * 1e9);
    }

    checkCudaError(cudaMemcpy(h_C, C, size, cudaMemcpyDeviceToHost), "Failed to copy C to host");

    std::cout << "GFLOPS: " << gflops << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(d_states);

    return 0;
}