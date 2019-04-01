#pragma once

#include <stdio.h>
#include <cuda_runtime_api.h>

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

inline size_t get_number_of_parts(size_t whole, size_t divider)
{
    return ((whole + divider - 1) / divider);
}

struct KernelSettings
{
    dim3 blockDimension;
    dim3 gridDimension;
};