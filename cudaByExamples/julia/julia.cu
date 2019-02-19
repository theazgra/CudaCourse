#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "../common/cpu_bitmap.h"

typedef unsigned char uchar;

constexpr int DIM = 1000;

/*
struct cuComplex
{
    float r;
    float i;
    cuComplex(float a, float b) : r(a), i(b)
    {
    }
    float magnitude2(void) { return r * r + i * i; }
    cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);

    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

void kernel(uchar *ptr)
{
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;
            int jul = julia(x, y);
            // RGBA
            ptr[offset * 4 + 0] = 255 * jul;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
}
*/

struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b)
    {
    }
    __device__ float magnitude2(void) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);

    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

__global__ void kernel(uchar *bmpPtr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int jul = julia(x, y);
    // RGBA
    bmpPtr[offset * 4 + 0] = 100 * jul;
    bmpPtr[offset * 4 + 1] = 50 * jul;
    bmpPtr[offset * 4 + 2] = 200 * jul;
    bmpPtr[offset * 4 + 3] = 255;
}

int main(void)
{
    CPUBitmap bmp(DIM, DIM);
    // CPU only computation
    /*
    uchar *ptr = bmp.get_ptr();
    kernel(ptr);
    bmp.display_and_exit();
    */

    // GPU computation
    uchar *device_bmp_ptr;
    cudaMalloc((void **)&device_bmp_ptr, bmp.image_size());

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(device_bmp_ptr);

    cudaMemcpy(bmp.get_ptr(), device_bmp_ptr, bmp.image_size(), cudaMemcpyDeviceToHost);
    bmp.display_and_exit();

    cudaFree(device_bmp_ptr);

    return 0;
}
