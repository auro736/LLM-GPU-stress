// filename: gpu_stress_test.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CHECK_CUDA(call) do {                                \
    cudaError_t err = call;                                  \
    if(err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA Error %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                  \
    }                                                       \
} while(0)

constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS = 128; // Tune for RTX 6000 Ada: 128 blocks * 256 threads = 32768 threads total
constexpr int MATRIX_DIM = 64; // 64x64 matrix multiply per thread block to stress FP units and shared memory
constexpr int SHARED_MEM_PER_BLOCK = MATRIX_DIM * MATRIX_DIM * sizeof(float) * 2; // For A and B tiles

// Kernel 1: Mixed precision matrix multiply with shared memory, stressing FP32 and FP64 units and L2 cache
// Each block computes a MATRIX_DIM x MATRIX_DIM tile of C
// Each thread computes one element of the tile
// Use float (FP32) for A and B, accumulate in double (FP64) to stress mixed precision math units
__global__ void matmul_stress_kernel(const float* __restrict__ A, const float* __restrict__ B, double* __restrict__ C, int dim) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = As + MATRIX_DIM * MATRIX_DIM;

    int tx = threadIdx.x % MATRIX_DIM;
    int ty = threadIdx.x / MATRIX_DIM;
    int row = blockIdx.y * MATRIX_DIM + ty;
    int col = blockIdx.x * MATRIX_DIM + tx;

    double acc = 0.0;

    for (int tile = 0; tile < dim / MATRIX_DIM; ++tile) {
        // Load A tile
        int arow = row;
        int acol = tile * MATRIX_DIM + tx;
        if (arow < dim && acol < dim)
            As[ty * MATRIX_DIM + tx] = A[arow * dim + acol];
        else
            As[ty * MATRIX_DIM + tx] = 0.0f;

        // Load B tile
        int brow = tile * MATRIX_DIM + ty;
        int bcol = col;
        if (brow < dim && bcol < dim)
            Bs[ty * MATRIX_DIM + tx] = B[brow * dim + bcol];
        else
            Bs[ty * MATRIX_DIM + tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        #pragma unroll 8
        for (int k = 0; k < MATRIX_DIM; ++k) {
            acc += static_cast<double>(As[ty * MATRIX_DIM + k]) * static_cast<double>(Bs[k * MATRIX_DIM + tx]);
        }

        __syncthreads();
    }

    if (row < dim && col < dim) {
        C[row * dim + col] = acc;
    }
}

// Kernel 2: Floating point special functions stress + atomic ops on shared memory
// Use __sinf, __expf, __logf, __cosf to stress XU units
// Atomic adds on shared memory to stress atomic units
__global__ void special_func_atomic_stress_kernel(float* output, int size) {
    extern __shared__ float shared_atomic[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % WARP_SIZE;

    // Initialize shared atomic memory once per block per lane
    if (lane == 0) {
        for (int i = 0; i < WARP_SIZE; ++i) {
            shared_atomic[i] = 0.0f;
        }
    }
    __syncthreads();

    if (tid < size) {
        float val = static_cast<float>(tid) * 0.001f + 1.0f;

        // Chain special functions to stress XU units
        float sf = __sinf(val);
        sf = __expf(sf);
        sf = __logf(sf + 1.0f);
        sf = __cosf(sf);

        // Atomic add on shared memory to stress atomic units
        atomicAdd(&shared_atomic[lane], sf);

        // Write back result to global memory to keep memory bandwidth busy
        output[tid] = sf + shared_atomic[lane];
    }
}

// Kernel 3: Intensive integer atomic operations on global memory
// Stress schedulers and atomic units with heavy atomicAdd on large array
__global__ void atomic_int_stress_kernel(int* data, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        int idx = tid % size;
        // Intense atomic add
        atomicAdd(&data[idx], 1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <duration_seconds>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int duration_seconds = atoi(argv[1]);
    if (duration_seconds <= 0) {
        fprintf(stderr, "Duration must be > 0\n");
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(0));

    // Matrix dimension for matmul kernel must be multiple of MATRIX_DIM
    constexpr int MAT_DIM = 4096; // 4096x4096 matrix to occupy large memory and L2 cache

    // Allocate matrices A, B (float), C (double)
    size_t matrix_size_float = MAT_DIM * MAT_DIM * sizeof(float);
    size_t matrix_size_double = MAT_DIM * MAT_DIM * sizeof(double);

    float* d_A = nullptr;
    float* d_B = nullptr;
    double* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, matrix_size_float));
    CHECK_CUDA(cudaMalloc(&d_B, matrix_size_float));
    CHECK_CUDA(cudaMalloc(&d_C, matrix_size_double));

    // Initialize A and B with some pattern (host side)
    float* h_A = new float[MAT_DIM * MAT_DIM];
    float* h_B = new float[MAT_DIM * MAT_DIM];
    for (int i = 0; i < MAT_DIM * MAT_DIM; ++i) {
        h_A[i] = static_cast<float>((i % 100) * 0.01f + 1.0f);
        h_B[i] = static_cast<float>(((i + 50) % 100) * 0.01f + 1.0f);
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_A, matrix_size_float, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, matrix_size_float, cudaMemcpyHostToDevice));
    delete[] h_A;
    delete[] h_B;

    // Allocate buffer for special_func_atomic_stress_kernel output
    constexpr int SPECIAL_SIZE = 16 * 1024 * 1024; // 16M floats ~64MB
    float* d_special_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_special_out, SPECIAL_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_special_out, 0, SPECIAL_SIZE * sizeof(float)));

    // Allocate buffer for atomic_int_stress_kernel
    constexpr int ATOMIC_INT_SIZE = 16 * 1024 * 1024; // 16M integers ~64MB
    int* d_atomic_int = nullptr;
    CHECK_CUDA(cudaMalloc(&d_atomic_int, ATOMIC_INT_SIZE * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_atomic_int, 0, ATOMIC_INT_SIZE * sizeof(int)));

    // Configure matmul grid and block dims
    dim3 block_dim_matmul(THREADS_PER_BLOCK, 1); // 256 threads in 1D
    dim3 grid_dim_matmul(MAT_DIM / MATRIX_DIM, MAT_DIM / MATRIX_DIM);

    // Configure special_func_atomic_stress_kernel grid and block dims
    int threads_special = THREADS_PER_BLOCK;
    int blocks_special = (SPECIAL_SIZE + threads_special - 1) / threads_special;

    // Configure atomic_int_stress_kernel grid and block dims
    int threads_atomic = THREADS_PER_BLOCK;
    int blocks_atomic = (ATOMIC_INT_SIZE + threads_atomic - 1) / threads_atomic;

    // Shared memory size for matmul kernel (2 tiles of MATRIX_DIM x MATRIX_DIM floats)
    size_t shared_mem_size_matmul = 2 * MATRIX_DIM * MATRIX_DIM * sizeof(float);

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    // Launch all kernels repeatedly until duration_seconds elapsed
    while (true) {
        matmul_stress_kernel<<<grid_dim_matmul, block_dim_matmul, shared_mem_size_matmul>>>(d_A, d_B, d_C, MAT_DIM);
        special_func_atomic_stress_kernel<<<blocks_special, threads_special, WARP_SIZE * sizeof(float)>>>(d_special_out, SPECIAL_SIZE);
        atomic_int_stress_kernel<<<blocks_atomic, threads_atomic>>>(d_atomic_int, ATOMIC_INT_SIZE);

        // Check all kernel launches
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (elapsed >= duration_seconds) break;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_special_out));
    CHECK_CUDA(cudaFree(d_atomic_int));

    printf("GPU stress test completed successfully for %d seconds.\n", duration_seconds);
    return EXIT_SUCCESS;
}