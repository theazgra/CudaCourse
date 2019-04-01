#include <cuda_utils.cuh>
#include <device_launch_parameters.h>
#include <image.h>
#include <assert.h>
#include <cell_grid.cuh>

#include <fitness_texture.cuh>

constexpr int NumberOfEvolutions = 500;

void load_fitness_image_into_texture(const char *imageFileName, ImageType imageType)
{
    assert(imageType == ImageType_GrayScale_8bpp && "Cuda texture only support 1,2 or 4 sized vectors.");

    Image img(imageFileName, ImageType_RGB_24bpp);

    size_t pitchedMemoryWidth = img.width() * sizeof(byte) * img.channel_count();
    size_t pitchedMemoryHeight = img.height();

    // HANDLE_ERROR(cudaMallocPitch((void **)&device_fitnessTextureData, &fitnessTexturePitch, pitchedMemoryWidth, pitchedMemoryHeight));
    // HANDLE_ERROR(cudaMemcpy2D(device_fitnessTextureData, fitnessTexturePitch, img.data(), pitchedMemoryWidth, pitchedMemoryWidth, img.height(), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMallocPitch((void **)&fitnessTexture.device_data, &fitnessTexture.memoryPitch, pitchedMemoryWidth, pitchedMemoryHeight));
    HANDLE_ERROR(cudaMemcpy2D(fitnessTexture.device_data, fitnessTexture.memoryPitch, img.data(), pitchedMemoryWidth, pitchedMemoryWidth, img.height(), cudaMemcpyHostToDevice));

    switch (imageType)
    {
    case ImageType_GrayScale_8bpp:
        //fitnessTextureCFD = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        fitnessTexture.textureCFD = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        break;
    case ImageType_RGB_24bpp:
        //fitnessTextureCFD = cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindUnsigned);
        fitnessTexture.textureCFD = cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindUnsigned);
        break;
    }

    fitnessTextureRef.normalized = false;
    fitnessTextureRef.filterMode = cudaFilterModePoint;
    fitnessTextureRef.addressMode[0] = cudaAddressModeClamp;
    fitnessTextureRef.addressMode[1] = cudaAddressModeClamp;

    //HANDLE_ERROR(cudaBindTexture2D(0, &fitnessTextureRef, device_fitnessTextureData, &fitnessTextureCFD, img.width(), img.height(), fitnessTexturePitch));
    HANDLE_ERROR(cudaBindTexture2D(0, &fitnessTextureRef, fitnessTexture.device_data, &fitnessTexture.textureCFD, img.width(), img.height(), fitnessTexture.memoryPitch));
}

__global__ void readTexIntoMem(byte *arr, const int width, const int height)
{
    int tidX = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tidY = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (tidX < width && tidY < height)
    {
        // Why is texture 0?
        //arr[(tidX * width) + tidY] = tex2D(fitnessTextureRef, tidX, tidY);
        arr[(tidX * width) + tidY] = tex2D(fitnessTextureRef, tidX, tidY);
    }
}

int main(int argc, char const *argv[])
{
    load_fitness_image_into_texture("../white.png", ImageType_GrayScale_8bpp);
    const int N = 10 * 10;

    byte *device_arr;
    HANDLE_ERROR(cudaMalloc((void **)&device_arr, N * sizeof(byte)));
    HANDLE_ERROR(cudaMemset(device_arr, 0, N * sizeof(byte)));
    KernelSettings ks = {};
    ks.gridDimension = dim3(1, 1, 1);
    ks.blockDimension = dim3(10, 10);
    readTexIntoMem<<<ks.gridDimension, ks.blockDimension>>>(device_arr, 10, 10);

    byte hostArr[N];
    HANDLE_ERROR(cudaMemcpy(hostArr, device_arr, N * sizeof(byte), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < 10; i++)
    {
        for (size_t j = 0; j < 10; j++)
        {
            printf("%u ", hostArr[(i * 10) + j]);
        }
        printf("\n");
    }

    HANDLE_ERROR(cudaFree(device_arr));

    // //TODO: Set kernel settings.
    // KernelSettings ks = {};
    // CellGrid grid(10, 10, ks);
    // grid.initialize_grid();

    // HANDLE_ERROR(cudaUnbindTexture(fitnessTextureRef));
    // HANDLE_ERROR(cudaFree(device_fitnessTextureData));
    return 0;
}
