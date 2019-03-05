#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <functional>
#include <math.h>
#include <time.h>
#include <random>

typedef unsigned char byte;

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                      \
    {                                                       \
        if (a == NULL)                                      \
        {                                                   \
            printf("Host memory failed in %s at line %d\n", \
                   __FILE__, __LINE__);                     \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    }

inline size_t get_number_of_parts(size_t whole, size_t divider)
{
    return ((whole + divider - 1) / divider);
}

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

// template <typename T>
// bool all_satisfy

template <typename T>
bool all_not_eq(const std::vector<T> &data, const T &cmp)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i] == cmp)
            return false;
    }
    return true;
}

struct KernelSetting
{
    dim3 dimGrid;
    dim3 dimBlock;
};
