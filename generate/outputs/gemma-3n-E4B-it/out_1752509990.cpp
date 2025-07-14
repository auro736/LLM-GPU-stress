#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_VERSION 11_0
#define MAX_BLOCK_SIZE 256

// Configuration parameters
#define ARRAY_SIZE 1024 // Size of matrices/arrays
#define NUM_ITERATIONS 1000
#define STRESS_INTENSITY 1.0f // Adjust to control workload intensity
#define HOST_DEVICE_TRANSFER_SIZE 1024 * 1024 // Size of data transferred between host and device
#define NUM_THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1
#define NUM_MATRIX_MULTIPLICATIONS 1

// Error handling macro
#define CUDA_CHECK(call)                                                                                                   \
    do {                                                                                                                     \
        cudaError_t err = call;                                                                                                \
        if (err != cudaSuccess) {                                                                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                                           \
        }                                                                                                                      \
    } while (0)

// Function to generate random floating-point numbers
std::vector<float> generate_random_array(size_t size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// CUDA kernel for trigonometric calculations
__global__ void trigonometric_calculations(float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = std::sin(input[i]) * std::cos(input[i]); //Example: sin(x) * cos(x)
    }
}

// CUDA kernel for atomic operations
__global__ void atomic_operations(int* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int value = atomicAdd(data, i, 1);
        atomicSubtract(data, i, 1);
    }
}

int main() {
    if (cudaError_t cudaStatus = cudaSetDevice(0); cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // Allocate host and device memory
    std::vector<float> host_A = generate_random_array(ARRAY_SIZE);
    std::vector<float> host_B = generate_random_array(ARRAY_SIZE);
    std::vector<float> host_C(ARRAY_SIZE * ARRAY_SIZE);
    float* device_A;
    float* device_B;
    float* device_C;

    CUDA_CHECK(cudaMalloc((void**)&device_A, ARRAY_SIZE * ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&device_B, ARRAY_SIZE * ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&device_C, ARRAY_SIZE * ARRAY_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(device_A, host_A.data(), ARRAY_SIZE * ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_B, host_B.data(), ARRAY_SIZE * ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));


    // Configure grid and block dimensions
    int blockSize = MAX_BLOCK_SIZE;
    int numBlocks = (ARRAY_SIZE / blockSize + (ARRAY_SIZE % blockSize > 0)) ;

    // Matrix Multiplication Stress Test
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        matrix_multiply<<<numBlocks, blockSize>>>(device_A, device_B, device_C, ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix Multiplication Time: " << duration.count() << " ms" << std::endl;

    //Calculate GFLOPS
    double size_in_bytes = ARRAY_SIZE * ARRAY_SIZE * sizeof(float);
    double iterations = NUM_ITERATIONS;
    double time_in_seconds = duration.count() / 1000.0;
    double gflops = (size_in_bytes * iterations) / (time_in_seconds * 1e9);
    std::cout << "Matrix Multiplication GFLOPS: " << gflops << std::endl;


    // Trigonometric Calculations Stress Test
     start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        trigonometric_calculations<<<numBlocks, blockSize>>>(device_A, device_C, ARRAY_SIZE * ARRAY_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Trigonometric Calculations Time: " << duration.count() << " ms" << std::endl;

    //Calculate GFLOPS
    gflops = (size_in_bytes * iterations) / (time_in_seconds * 1e9);
    std::cout << "Trigonometric Calculations GFLOPS: " << gflops << std::endl;


    // Atomic Operations Stress Test

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        atomic_operations<<<numBlocks, blockSize>>>(device_C, ARRAY_SIZE * ARRAY_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Atomic Operations Time: " << duration.count() << " ms" << std::endl;

    //Calculate GFLOPS
    gflops = (size_in_bytes * iterations) / (time_in_seconds * 1e9);
    std::cout << "Atomic Operations GFLOPS: " << gflops << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(host_C.data(), device_C, ARRAY_SIZE * ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));
    CUDA_CHECK(cudaFree(device_C));

    return 0;
}