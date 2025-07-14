#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

// Define constants
#define ARRAY_SIZE 1024 * 1024 * 1024 // 1 GB
#define NUM_THREADS 256
#define BLOCK_SIZE 32
#define NUM_BLOCKS (ARRAY_SIZE / (NUM_THREADS / BLOCK_SIZE))

// Define CUDA error checking macro
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                   \
        cudaError_t err = call;                                                             \
        if (err != cudaSuccess) {                                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;     \
            exit(1);                                                                        \
        }                                                                                 \
    } while (0)

// Function to generate random floating-point numbers
std::vector<float> generate_random_data(size_t size) {
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
__global__ void matrix_multiply(float *A, float *B, float *C, int rows, int cols, int block_size) {
    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;

    if (row < rows && col < cols) {
        float sum = 0.0f;
        for (int k = 0; k < cols; ++k) {
            sum += A[row * cols + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

// CUDA kernel for trigonometric operations
__global__ void trigonometric_kernel(float *data, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = std::sin(data[idx]) * std::cos(data[idx]);
    }
}

// CUDA kernel for atomic operations
__global__ void atomic_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(data, 1, idx);
    }
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <stress_level> <duration_ms> <workload>" << std::endl;
        std::cerr << "  stress_level: 1 (low), 2 (medium), 3 (high)" << std::endl;
        std::cerr << "  duration_ms: Test duration in milliseconds" << std::endl;
        std::cerr << "  workload: matrix, trig, atomic" << std::endl;
        return 1;
    }

    int stress_level = std::stoi(argv[1]);
    int duration_ms = std::stoi(argv[2]);
    std::string workload = argv[3];

    if (workload != "matrix" && workload != "trig" && workload != "atomic") {
        std::cerr << "Invalid workload. Choose from matrix, trig, or atomic." << std::endl;
        return 1;
    }
    
    size_t data_size = ARRAY_SIZE;
    float *host_data = nullptr;
    float *device_data = nullptr;
    float *device_result = nullptr;
    float *device_atomic_data = nullptr;

    // Allocate host memory
    host_data = new float[data_size];
    device_data = (float*)cudaMalloc(&device_data, data_size * sizeof(float));
    device_result = (float*)cudaMalloc(&device_result, data_size * sizeof(float));
    device_atomic_data = (float*)cudaMalloc(&device_atomic_data, data_size * sizeof(float));

    // Initialize host data
    std::vector<float> data = generate_random_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        host_data[i] = data[i];
    }

    // --- Matrix Multiplication ---
    if (workload == "matrix") {
        // Configure block size
        int blockSize = BLOCK_SIZE;
        int numBlocks = (data_size / (NUM_THREADS / blockSize));

        // Launch kernel
        matrix_multiply<<<numBlocks, NUM_THREADS>>>(device_data, device_data, device_result, data_size / 2, data_size / 2, blockSize);
        cudaDeviceSynchronize();
    }
    // --- Trigonometric Operations ---
    else if (workload == "trig") {
        //Configure block size
         int blockSize = BLOCK_SIZE;
         int numBlocks = (data_size / (NUM_THREADS / blockSize));

        // Launch kernel
        trigonometric_kernel<<<numBlocks, NUM_THREADS>>>(device_data, device_result, data_size);
        cudaDeviceSynchronize();
    }
    // --- Atomic Operations ---
    else if (workload == "atomic") {
        //Configure block size
         int blockSize = BLOCK_SIZE;
         int numBlocks = (data_size / (NUM_THREADS / blockSize));
        
        // Launch kernel
        atomic_kernel<<<numBlocks, NUM_THREADS>>>(device_data, data_size);
        cudaDeviceSynchronize();
    }

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - start < std::chrono::milliseconds(duration_ms)) {
       //Do nothing to allow kernel to execute.
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Calculate GFLOPS/TOPS (approximate)
    double gflops = (double)data_size * 1000 * 1.0 / (duration.count() / 1000.0) / 1e9; // GFLOPS
    std::cout << "GFLOPS: " << gflops << std::endl;

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_data);
        cudaFree(device_result);
        cudaFree(device_atomic_data);
        delete host_data;
        return 1;
    }
    

    // Free device memory
    cudaFree(device_data);
    cudaFree(device_result);
    cudaFree(device_atomic_data);
    delete host_data;

    std::cout << "Stress test completed." << std::endl;
    return 0;
}