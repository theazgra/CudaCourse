#pragma once
#include <cassert>
#include <stdio.h>
#include <cuda_runtime_api.h>

typedef unsigned char byte;
typedef unsigned int uint;

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(1);
    }
}
static void HandleCurandError(int err, const char *file, int line)
{
    if (err != 0) //CURAND_STATUS_SUCCESS = 0
    {
        printf("Currant error at %s:%d\n", file, line);
        exit(1);
    }
}

#define CUDA_CALL(err) (HandleError(err, __FILE__, __LINE__))
#define CURAND_CALL(err) (HandleCurandError(err, __FILE__, __LINE__))

#define CUDA_TIMED_BLOCK_START(fn_name) \
    const char *___tmdFnName = fn_name; \
    cudaEvent_t startEvent, stopEvent;  \
    float elapsedTime;                  \
    cudaEventCreate(&startEvent);       \
    cudaEventCreate(&stopEvent);        \
    cudaEventRecord(startEvent, 0);

#define CUDA_TIMED_BLOCK_END                                   \
    cudaEventRecord(stopEvent, 0);                             \
    cudaEventSynchronize(stopEvent);                           \
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent); \
    printf("%s took: %f ms\n", ___tmdFnName, elapsedTime);     \
    cudaEventDestroy(startEvent);                              \
    cudaEventDestroy(stopEvent);

inline size_t get_number_of_parts(size_t whole, size_t divider)
{
    return ((whole + divider - 1) / divider);
}

struct KernelSettings
{
    dim3 blockDimension;
    dim3 gridDimension;
};

template <typename T>
__host__ void print_device_memory(const T *device_memory, size_t pitch, size_t width, size_t height, const char *format)
{
    if (device_memory == nullptr)
        return;

    T *tmpMemory;
    CUDA_CALL(cudaHostAlloc((void **)&tmpMemory, pitch * height, cudaHostAllocWriteCombined));
    CUDA_CALL(cudaMemcpy(tmpMemory, device_memory, pitch * height, cudaMemcpyDeviceToHost));

    T *dataPtr = tmpMemory;
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            printf(format, dataPtr[col]);
        }
        printf("\n");
        dataPtr = (T *)(((char *)dataPtr) + pitch);
    }

    CUDA_CALL(cudaFreeHost(tmpMemory));
}