#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define GRID_SIZE_X 1024
#define GRID_SIZE_Y 1024

__global__ void stressTestKernel(float *d_input, float *d_output, int width, int height, int duration) {
    extern __shared__ float s_data[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    curandState state;
    curand_init(clock64(), x * width + y, 0, &state);

    for (int t = 0; t < duration; ++t) {
        float a = d_input[y * width + x];
        float b = curand_uniform(&state);
        float c = sinf(a) + cosf(b);
        float d = a * b + c / (CUDART_INF + 0.01f);
        atomicAdd(&d_output[y * width + x], d);
        sum += d;
    }

    s_data[threadIdx.y * blockDim.x + threadIdx.x] = sum;

    __syncthreads();

    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.y * blockDim.x + threadIdx.x] += s_data[threadIdx.y * blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&d_output[blockIdx.y * gridDim.x + blockIdx.x], s_data[0]);
    }
}

int main(int argc, char **argv) {
    int width = 16384;
    int height = 16384;
    int duration = 1000;
    size_t size = width * height * sizeof(float);

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemset(d_output, 0, size);

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_input, width * height);
    curandDestroyGenerator(gen);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    stressTestKernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, width, height, duration);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float *h_output = (float *)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Stress test completed in " << milliseconds << " ms" << std::endl;

    return 0;
}