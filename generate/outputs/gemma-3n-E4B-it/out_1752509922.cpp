#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// Configuration parameters
const int MATRIX_SIZE = 1024; // Size of matrices
const int NUM_ITERATIONS = 100; // Number of iterations for each test
const float STRESS_INTENSITY = 1.0f; // Scaling factor for workload intensity
const int BLOCK_SIZE = 256; // Block size for CUDA kernels
const int GRID_SIZE = MATRIX_SIZE / BLOCK_SIZE; // Grid size for CUDA kernels

// CUDA Compute Capability Check
#define CUDA_COMPUTE_CAPABILITY 7

// Error handling macro
#define CUDA_CHECK(call)                                         \
    do {                                                           \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                \
        }                                                         \
    } while (0)

// Function to generate random matrices
std::vector<float> generate_random_matrix(int size) {
    std::vector<float> matrix(size * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);

    for (int i = 0; i < size * size; ++i) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(float* A, float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; ++k) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

// CUDA kernel for trigonometric calculations
__global__ void trigonometric_calculations(float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = std::sin(input[i]) + std::cos(input[i]) + std::tan(input[i]);
    }
}

// CUDA kernel for atomic operations
__global__ void atomic_operations(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(data, data[i]);
    }
}

int main() {
    if (cudaGetDeviceCount() < 1) {
        std::cerr << "No CUDA-capable device found." << std::endl;
        return 1;
    }

    int device_id = 0; // Use the first GPU

    // Allocate host memory
    std::vector<float> h_A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> h_B(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> h_C(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> h_input(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> h_output(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> h_data(MATRIX_SIZE * MATRIX_SIZE);



    // Allocate device memory
    float* d_A, * d_B, * d_C, * d_input, * d_output, * d_data;
    CUDA_CHECK(cudaMalloc(&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));


    // Initialize matrices on the host
    h_A = generate_random_matrix(MATRIX_SIZE);
    h_B = generate_random_matrix(MATRIX_SIZE);
    h_input = generate_random_matrix(MATRIX_SIZE);
    h_data = generate_random_matrix(MATRIX_SIZE);



    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));


    // Performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Run matrix multiplication
    matrix_multiply<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, MATRIX_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Run trigonometric calculations
    trigonometric_calculations<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, MATRIX_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Run atomic operations
    atomic_operations<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, MATRIX_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Calculate and print metrics
    double elapsed_seconds = duration.count() / 1000.0;

    // Calculate GFLOPS/TOPS
    double gflops = (double)(MATRIX_SIZE * MATRIX_SIZE * NUM_ITERATIONS) * STRESS_INTENSITY / (elapsed_seconds * 1e9);
    double flops = (double)(MATRIX_SIZE * MATRIX_SIZE* NUM_ITERATIONS) / (elapsed_seconds * 1e9);

    // Calculate memory bandwidth utilization (approximate)
    double memory_bandwidth = (double)(MATRIX_SIZE * MATRIX_SIZE * sizeof(float) * NUM_ITERATIONS) / (elapsed_seconds * 1e6);

    // Calculate GPU occupancy (approximate)
    double occupancy = (double)(BLOCK_SIZE * GRID_SIZE) / (1024 * 1024); // Number of warps per block

    std::cout << "Test Results:" << std::endl;
    std::cout << "Elapsed Time: " << elapsed_seconds << " seconds" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;
    std::cout << "TOPS: " << flops << std::endl;
    std::cout << "Memory Bandwidth: " << memory_bandwidth << " MB/s" << std::endl;
    std::cout << "GPU Occupancy: " << occupancy << std::endl;


    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA