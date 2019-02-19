#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int N = 55;
constexpr int M = 3;

__global__ void simplest_kernel_sum(int *A, int *B, int *result)
{
    for (size_t i = 0; i < N; i++)
    {
        result[i] = A[i] + B[i];
    }
}

__global__ void parallel_kernel_sum(int *A, int *B, int *result)
{
    int i = blockIdx.x;
    if (i < N)
        result[i] = A[i] + B[i];
}

__global__ void kernel_sum_pairs(int *data, int *result)
{
    int m = blockIdx.x;
    if (m < M)
    {
        result[m] = 0;
        for (size_t n = 0; n < N; n++)
        {
            result[m] += data[n + m];
        }
    }
}

int main(void)
{
#if 0
    
    /*
    // Allocate the HOST memory that will represent two M-dimensional vectors (A, B) and fill them with some values.
    int A[N];
    int B[N];

    for (size_t i = 0; i < N; i++)
    {
        A[i] = i * i - N;
        B[i] = i + i / i;
    }

    // Allocate the DEVICE memory to be able to copy data from HOST.
    int *device_A;
    int *device_B;
    int *device_C;
    cudaMalloc((void **)&device_A, sizeof(int) * N);
    cudaMalloc((void **)&device_B, sizeof(int) * N);
    // Allocate the DEVICE memory to store an output M-dimensional vector C.
    cudaMalloc((void **)&device_C, sizeof(int) * N);

    cudaMemcpy(device_A, A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Create a kernel that sums scalar values such that C[i] = A[i] + B[i].
    //simplest_kernel_sum<<<1, 1>>>(device_A, device_B, device_C);
    parallel_kernel_sum<<<N, 1>>>(device_A, device_B, device_C);

    int C[N];
    cudaMemcpy(C, device_C, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i++)
    {
        printf("%i + %i = %i\n", A[i], B[i], C[i]);
    }

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    */
#endif

    // Allocate the HOST memory that will represent N M-dimensional vectors (A_0,...A_n-1, B_0, ... B_n-1) and fill them with some values.
    int data[N][M];
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            data[i][j] = i + j;
        }
    }

    // Allocate the DEVICE memory to be able to copy data from HOST.
    int *device_data;
    cudaMalloc((void **)&device_data, sizeof(int) * N * M);

    cudaMemcpy(device_data, data, sizeof(int) * N * M, cudaMemcpyHostToDevice);

    // Allocate the DEVICE memory to store output M-dimensional vectors C_0 ... C_n-1.
    int *device_result;
    cudaMalloc((void **)&device_result, sizeof(int) * M);

    // Create a kernel that sums all vectors pairs that C_0[i] = A_0[i] + B_0[i], ... C_n-1[i] = A_n-1[i] + B_n-1[i].
    // THINK ABOUT THE VARIANTS OF YOUR SOLUTION, CONSIDER THE PROS AND CONS.
    kernel_sum_pairs<<<M, 1>>>(device_data, device_result);

    // 2-D kernel can only be used if we synchronize the addition.

    int result[M];
    cudaMemcpy(result, device_result, sizeof(int) * M, cudaMemcpyDeviceToHost);

    for (size_t m = 0; m < M; m++)
    {
        printf("result[%i] = %i\n", (int)m, result[m]);
    }

    cudaFree(device_data);
    cudaFree(device_result);

    return 0;
}