#include "my_cuda.cu"
#include "common/cpu_anim.h"
constexpr int N = 1024 * 33;
constexpr int DIM = 1024;

struct DataBlock
{
    byte *device_bitmap;
    CPUAnimBitmap *bitmap;
};

void cleanup(DataBlock *db)
{
    cudaFree(db->device_bitmap);
}

__global__ void add(int *a, int *b, int *c)
{
    //int threadId = threadIdx.x;
    //int threadId = blockIdx.x;
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    while (threadId < N)
    {
        c[threadId] = a[threadId] + b[threadId];
        threadId += blockDim.x * gridDim.x;
    }
}

__global__ void kernel(byte *ptr, int ticks)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int offset = (blockDim.x * gridDim.x * y) + x;

    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);

    byte gray = (byte)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
    //float grayF = (128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = gray;
    ptr[offset * 4 + 3] = 255;
}

void generate_frame(DataBlock *db, int ticks)
{
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(db->device_bitmap, ticks);

    cudaMemcpy(db->bitmap->get_ptr(), db->device_bitmap,
               db->bitmap->image_size(), cudaMemcpyDeviceToHost);
}

int main(int, char **)
{
    /*
    int a[N];
    int b[N];
    int *device_a, *device_b, *device_c;

    HANDLE_ERROR(cudaMalloc((void **)&device_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&device_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&device_c, N * sizeof(int)));

    for (size_t i = 0; i < N; i++)
    {
        a[i] = i * i;
        b[i] = i + i;
    }

    HANDLE_ERROR(cudaMemcpy(device_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(device_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    // Calling N blocks.
    //add<<<N, 1>>>(device_a, device_b, device_c);
    // Calling N threads.
    add<<<128, 128>>>(device_a, device_b, device_c);

    int c[N];
    HANDLE_ERROR(cudaMemcpy(c, device_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (size_t i = N - 100; i < N; i++)
    {
        printf("%i + %i = %i\n", a[i], b[i], c[i]);
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    */
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    cudaMalloc((void **)&data.device_bitmap, bitmap.image_size());

    bitmap.anim_and_exit((void (*)(void *, int))generate_frame, (void (*)(void *))cleanup);
    return 0;
}