//#include <cuda.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main(void)
{
    int c;
    int *device_c;

    cudaMalloc((void **)&device_c, sizeof(int));
    add<<<1, 1>>>(2, 7, device_c);
    cudaMemcpy(&c, device_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_c);
    printf("2 + 7 = %i\nComputed via CUDA, yaaay.\n", c);

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    printf("This computer contains %i CUDA enabled GPU.\n", cudaDeviceCount);

    std::vector<cudaDeviceProp> deviceInfos;
    for (size_t i = 0; i < cudaDeviceCount; i++)
    {
        cudaDeviceProp info = {};
        cudaGetDeviceProperties(&info, i);
        deviceInfos.push_back(info);
        printf("Loaded info about %s\n", info.name);
    }

    // We can request device with certain capabilities like this:
    cudaDeviceProp requirement;
    memset(&requirement, 0, sizeof(cudaDeviceProp));
    // Request version 6.0
    requirement.major = 6;
    requirement.minor = 0;

    int returnedDevice;
    HANDLE_ERROR(cudaChooseDevice(&returnedDevice, &requirement));
    printf("cudaChooseDevice returned: %i\n", returnedDevice);
    HANDLE_ERROR(cudaSetDevice(returnedDevice));
    return 0;
}