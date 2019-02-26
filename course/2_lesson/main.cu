#include "../../my_cuda.cu"
#include "Array2D.h"

constexpr int N = 300000;
constexpr int M = 15;

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
            result[m] += data[(n * M) + m];
        }
    }
}

constexpr uint THREADS_PER_BLOCK = 256;
//constexpr uint MEMBLOCKS_PER_THREADBLOCK = 2;

__global__ void kernel_add_l3(int *a, int *b, int *c)
{
    uint offset = (blockIdx.x * THREADS_PER_BLOCK) + threadIdx.x;
    uint skip = gridDim.x * THREADS_PER_BLOCK;
    while (offset < N)
    {
        c[offset] = a[offset] + b[offset];
        offset += skip;
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

    /*
    // Allocate the HOST memory that will represent N M-dimensional vectors (A_0,...A_n-1, B_0, ... B_n-1) and fill them with some values.
    Array2D<int> data(N, M);
    //int data[N][M];
    for (size_t row = 0; row < N; row++)
    {
        for (size_t col = 0; col < M; col++)
        {
            data.at(row, col) = row * col + row;
        }
    }
    data.print();

    // Allocate the DEVICE memory to be able to copy data from HOST.
    int *device_data;
    cudaMalloc((void **)&device_data, sizeof(int) * N * M);

    cudaMemcpy(device_data, data.data_ptr(), sizeof(int) * N * M, cudaMemcpyHostToDevice);

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
    */

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
    cudaMalloc((void **)&device_C, sizeof(int) * N);

    cudaMemcpy(device_A, A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Create a kernel that sums scalar values such that C[i] = A[i] + B[i].
    //simplest_kernel_sum<<<1, 1>>>(device_A, device_B, device_C);
    cudaEvent_t startEvent, stopEvent;
    float elapsedTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    //parallel_kernel_sum<<<N, 1>>>(device_A, device_B, device_C);
    kernel_add_l3<<<get_number_of_parts(N, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(device_A, device_B, device_C);
    //someKernel<<<grids, blocks, 0, 0>>>(...);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    printf("Time to get device properties: %f ms\n", elapsedTime);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    int C[N];
    cudaMemcpy(C, device_C, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < N; i++)
    // {
    //     printf("%i + %i = %i\n", A[i], B[i], C[i]);
    // }

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    return 0;
}
