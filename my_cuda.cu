#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <math.h>

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