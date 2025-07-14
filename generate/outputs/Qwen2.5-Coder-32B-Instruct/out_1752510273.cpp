#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <curand_kernel.h>

#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE 1024
#define MAX_ITERATIONS 1000000

__constant__ float const_scalar;

// Kernel for matrix multiplication with shared memory
__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (N-1)/BLOCK_SIZE + 1; ++m) {
        if (row < N && m*BLOCK_SIZE + tx < N)
            s_A[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && m*BLOCK_SIZE + ty < N)
            s_B[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Cvalue += s_A[ty][k] * s_B[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

// Kernel for random memory access pattern
__global__ void randomMemoryAccess(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = sinf(data[idx]) + cosf(data[idx]) + __logf(data[idx] + 1.0f);
    }
}

// Kernel for atomic operations
__global__ void atomicOperations(int* sharedCounter, int* globalCounter, int size) {
    extern __shared__ int s_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(globalCounter, 1);
        atomicAdd(&s_data[threadIdx.x], 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(sharedCounter, s_data[0]);
        }
    }
}

// Kernel for streaming memory accesses
__global__ void streamingMemoryAccess(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * const_scalar + sinf(data[idx]);
    }
}

// Function to initialize random data
void initializeRandomData(float* data, int size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, data, size);
    curandDestroyGenerator(gen);
}

// Function to initialize constant memory
void initializeConstantMemory(float scalar) {
    cudaMemcpyToSymbol(const_scalar, &scalar, sizeof(float));
}

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << err << ") - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int N = 4096; // Matrix size
    int iterations = 1000; // Number of iterations
    float scalar = 2.5f; // Scalar for streaming memory access
    int size = N * N; // Total size of matrix

    // Allocate host memory
    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C = new float[size];

    // Initialize host memory
    initializeRandomData(h_A, size);
    initializeRandomData(h_B, size);
    initializeRandomData(h_C, size);

    // Allocate device memory
    float* d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, size * sizeof(float)), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, size * sizeof(float)), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, size * sizeof(float)), "cudaMalloc d_C");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_B");

    // Initialize constant memory
    initializeConstantMemory(scalar);

    // Set up dimensions for matrix multiplication
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Set up dimensions for random memory access
    int blockSizeRM = 256;
    int gridSizeRM = (size + blockSizeRM - 1) / blockSizeRM;

    // Set up dimensions for atomic operations
    int blockSizeAO = 256;
    int gridSizeAO = (size + blockSizeAO - 1) / blockSizeAO;

    // Set up dimensions for streaming memory access
    int blockSizeSA = 256;
    int gridSizeSA = (size + blockSizeSA - 1) / blockSizeSA;

    // Allocate shared memory for atomic operations
    int sharedMemSizeAO = blockSizeAO * sizeof(int);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // Matrix multiplication
        matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        checkCudaError(cudaGetLastError(), "matrixMulShared");

        // Random memory access
        randomMemoryAccess<<<gridSizeRM, blockSizeRM>>>(d_C, size);
        checkCudaError(cudaGetLastError(), "randomMemoryAccess");

        // Atomic operations
        int* d_sharedCounter, *d_globalCounter;
        checkCudaError(cudaMalloc((void**)&d_sharedCounter, sizeof(int)), "cudaMalloc d_sharedCounter");
        checkCudaError(cudaMalloc((void**)&d_globalCounter, sizeof(int)), "cudaMalloc d_globalCounter");
        checkCudaError(cudaMemset(d_sharedCounter, 0, sizeof(int)), "cudaMemset d_sharedCounter");
        checkCudaError(cudaMemset(d_globalCounter, 0, sizeof(int)), "cudaMemset d_globalCounter");
        atomicOperations<<<gridSizeAO, blockSizeAO, sharedMemSizeAO>>>(d_sharedCounter, d_globalCounter, size);
        checkCudaError(cudaGetLastError(), "atomicOperations");

        // Streaming memory access
        streamingMemoryAccess<<<gridSizeSA, blockSizeSA>>>(d_C, size);
        checkCudaError(cudaGetLastError(), "streamingMemoryAccess");

        // Free temporary device memory
        checkCudaError(cudaFree(d_sharedCounter), "cudaFree d_sharedCounter");
        checkCudaError(cudaFree(d_globalCounter), "cudaFree d_globalCounter");
    }

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_C");

    // Calculate performance metrics
    double gflops = 2.0 * N * N * N * iterations / duration.count() / 1e9;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Free device memory
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_B), "cudaFree d_B");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
```

This CUDA program is designed to stress test modern GPUs by combining intensive mathematical operations like matrix multiplication, random memory access, atomic operations, and streaming memory accesses. It uses shared memory, constant memory, and efficient memory access patterns to maximize resource utilization. The program includes comprehensive error handling and calculates performance metrics such as GFLOPS. It is optimized for CUDA Compute Capability 7.0+ GPUs with 16GB+ VRAM, targeting architectures like the RTX 4090, Tesla V100, and A100 series.