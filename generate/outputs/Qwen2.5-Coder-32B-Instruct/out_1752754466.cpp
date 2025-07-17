#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#define BLOCK_SIZE 256
#define TILE_SIZE 32

__device__ __inline__ float atomicAdd(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void stressKernel(float* a, float* b, float* c, float* sharedSum, int N) {
    extern __shared__ float sharedData[];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;

    for (int m = 0; m < N / TILE_SIZE; ++m) {
        if (row < N && m * TILE_SIZE + tx < N) {
            sharedData[ty * TILE_SIZE + tx] = a[row * N + m * TILE_SIZE + tx];
        } else {
            sharedData[ty * TILE_SIZE + tx] = 0.0f;
        }
        if (col < N && m * TILE_SIZE + ty < N) {
            sharedData[(TILE_SIZE + ty) * TILE_SIZE + tx] = b[(m * TILE_SIZE + ty) * N + col];
        } else {
            sharedData[(TILE_SIZE + ty) * TILE_SIZE + tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedData[ty * TILE_SIZE + k] * sharedData[(TILE_SIZE + k) * TILE_SIZE + tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        atomicAdd(&c[row * N + col], sum);
    }

    // Stress XU units with transcendental functions
    float stressVal = sinf((float)(row * col)) + cosf((float)(row + col)) + logf((float)(row + col + 1));
    atomicAdd(&sharedSum[tx], stressVal);
}

int main(int argc, char** argv) {
    int N = 4096; // Matrix dimension
    int duration = 10; // Test duration in seconds

    if (argc > 1) {
        N = std::stoi(argv[1]);
    }
    if (argc > 2) {
        duration = std::stoi(argv[2]);
    }

    float* h_a, *h_b, *h_c;
    float* d_a, *d_b, *d_c, *d_sharedSum;

    size_t bytes = N * N * sizeof(float);
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_sharedSum, BLOCK_SIZE * sizeof(float));

    for (int i = 0; i < N * N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, bytes);
    cudaMemset(d_sharedSum, 0, BLOCK_SIZE * sizeof(float));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        stressKernel<<<dimGrid, dimBlock, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_a, d_b, d_c, d_sharedSum, N);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (elapsed.count() >= duration) {
            break;
        }
    }

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a, d_sharedSum, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum += h_a[i];
    }

    std::cout << "Sum of stress values: " << sum << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_sharedSum);

    return 0;
}