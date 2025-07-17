// gpu_stress_test.cu

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;

// Define constants
const int MATRIX_SIZE = 1024;
const int NUM_THREADS = 256;
const int NUM_BLOCKS = 256;
const int MEM_ACCESS_PATTERN = 16; // Access pattern to maximize L2 cache usage

// Define kernel function to stress computational units
__global__ void stress_compute(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += sinf(data[idx] * 3.14159f);
        sum += cosf(data[idx] * 3.14159f);
        sum += expf(data[idx]);
        sum += logf(data[idx]);
    }
    data[idx] = sum;
}

// Define kernel function to stress memory
__global__ void stress_memory(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += data[idx + i * MEM_ACCESS_PATTERN];
    }
    data[idx] = sum;
}

// Define kernel function to stress schedulers and atomic operations
__global__ void stress_scheduler(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&sum, data[idx]);
    }
    data[idx] = sum;
}

// Define kernel function to stress matrix multiplication
__global__ void stress_matrix_multiply(float *A, float *B, float *C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        sum += A[row * MATRIX_SIZE + i] * B[i * MATRIX_SIZE + col];
    }
    C[row * MATRIX_SIZE + col] = sum;
}

int main(int argc, char **argv) {
    // Check for user-defined test duration
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <test_duration_in_seconds>" << endl;
        return 1;
    }
    int test_duration = atoi(argv[1]);

    // Initialize CUDA devices
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }

    // Allocate memory on the GPU
    float *data, *A, *B, *C;
    cudaMalloc((void **)&data, NUM_BLOCKS * NUM_THREADS * sizeof(float));
    cudaMalloc((void **)&A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&C, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize data
    float *h_data, *h_A, *h_B;
    h_data = new float[NUM_BLOCKS * NUM_THREADS];
    h_A = new float[MATRIX_SIZE * MATRIX_SIZE];
    h_B = new float[MATRIX_SIZE * MATRIX_SIZE];
    for (int i = 0; i < NUM_BLOCKS * NUM_THREADS; i++) {
        h_data[i] = i * 1.0f;
    }
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 1.0f;
    }
    cudaMemcpy(data, h_data, NUM_BLOCKS * NUM_THREADS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    auto start_time = chrono::high_resolution_clock::now();
    while (true) {
        stress_compute<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_memory<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_scheduler<<<NUM_BLOCKS, NUM_THREADS>>>(data);
        stress_matrix_multiply<<<dim3(NUM_BLOCKS, NUM_BLOCKS), dim3(NUM_THREADS, NUM_THREADS)>>> (A, B, C);
        //cudaDeviceSynchronize();
        auto current_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
        if (duration > test_duration) {
            break;
        }
    }

    // Clean up
    cudaFree(data);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] h_data;
    delete[] h_A;
    delete[] h_B;

    return 0;
}