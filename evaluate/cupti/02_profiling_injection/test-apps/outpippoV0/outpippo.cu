// stress_rtx20.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

#define TILE 32
#define N 1024
#define M 1024
#define K 1024
#define NUM_STREAMS 4
#define DURATION_SECONDS 30

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void matMulKernel(const float *A, const float *B, float *C,
                             unsigned long long *counter)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aIdx = row * K + t * TILE + threadIdx.x;
        int bIdx = (t * TILE + threadIdx.y) * K + col;
        sA[threadIdx.y][threadIdx.x] = (row < N && t * TILE + threadIdx.x < K) ? __ldg(&A[aIdx]) : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (t * TILE + threadIdx.y < K && col < K) ? __ldg(&B[bIdx]) : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < K) {
        int idx = row * K + col;
        C[idx] = sum;
        float val = sinf(sum) * cosf(sum) + expf(sum);
        atomicAdd(counter, *(unsigned long long *)&val);
    }
}

int main()
{
    size_t sizeA = N * K * sizeof(float);
    size_t sizeB = K * M * sizeof(float);
    size_t sizeC = N * M * sizeof(float);

    float *hA = (float *)malloc(sizeA);
    float *hB = (float *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);

    for (int i = 0; i < N * K; ++i) hA[i] = (float)(rand()) / RAND_MAX;
    for (int i = 0; i < K * M; ++i) hB[i] = (float)(rand()) / RAND_MAX;

    float *dA, *dB, *dC;
    unsigned long long *dCounter;
    CHECK_CUDA(cudaMalloc((void **)&dA, sizeA));
    CHECK_CUDA(cudaMalloc((void **)&dB, sizeB));
    CHECK_CUDA(cudaMalloc((void **)&dC, sizeC));
    CHECK_CUDA(cudaMalloc((void **)&dCounter, sizeof(unsigned long long)));

    CHECK_CUDA(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC));
    CHECK_CUDA(cudaMemset(dCounter, 0, sizeof(unsigned long long)));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    auto start = std::chrono::steady_clock::now();
    while (true) {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            matMulKernel<<<gridDim, blockDim, 0, streams[i]>>>(dA, dB, dC, dCounter);
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= DURATION_SECONDS)
            break;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned long long hostCounter;
    CHECK_CUDA(cuda