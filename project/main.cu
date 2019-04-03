#include <cuda_utils.cuh>
#include <device_launch_parameters.h>
#include <image.h>
#include <assert.h>
#include <cell_grid.cuh>

#include <fitness_texture.cuh>

constexpr int NumberOfEvolutions = 500;
constexpr int BlockDimensionSize = 16;
constexpr int CellGridDimension = 500;

void load_fitness_image_into_texture(const char *imageFileName, ImageType imageType)
{
    assert(imageType == ImageType_GrayScale_8bpp && "Cuda texture only support 1,2 or 4 sized vectors.");

    Image img(imageFileName, ImageType_RGB_24bpp);

    size_t pitchedMemoryWidth = img.width() * sizeof(byte) * img.channel_count();
    size_t pitchedMemoryHeight = img.height();

    // CUDA_CALL(cudaMallocPitch((void **)&device_fitnessTextureData, &fitnessTexturePitch, pitchedMemoryWidth, pitchedMemoryHeight));
    // CUDA_CALL(cudaMemcpy2D(device_fitnessTextureData, fitnessTexturePitch, img.data(), pitchedMemoryWidth, pitchedMemoryWidth, img.height(), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMallocPitch((void **)&fitnessTexture.device_data, &fitnessTexture.memoryPitch, pitchedMemoryWidth, pitchedMemoryHeight));
    CUDA_CALL(cudaMemcpy2D(fitnessTexture.device_data, fitnessTexture.memoryPitch, img.data(), pitchedMemoryWidth, pitchedMemoryWidth, img.height(), cudaMemcpyHostToDevice));

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

    //CUDA_CALL(cudaBindTexture2D(0, &fitnessTextureRef, device_fitnessTextureData, &fitnessTextureCFD, img.width(), img.height(), fitnessTexturePitch));
    CUDA_CALL(cudaBindTexture2D(0, &fitnessTextureRef, fitnessTexture.device_data, &fitnessTexture.textureCFD, img.width(), img.height(), fitnessTexture.memoryPitch));
}

int main(int argc, char const *argv[])
{
    //load_fitness_image_into_texture("../white.png", ImageType_GrayScale_8bpp);
    KernelSettings ks = {};
    ks.blockDimension = dim3(BlockDimensionSize, BlockDimensionSize, 1);
    ks.gridDimension = dim3(get_number_of_parts(CellGridDimension, BlockDimensionSize), get_number_of_parts(CellGridDimension, BlockDimensionSize), 1);

    CellGrid grid(CellGridDimension, CellGridDimension, ks);
    grid.initialize_grid();

    // for (size_t evolutionStep = 0; evolutionStep < NumberOfEvolutions; evolutionStep++)
    // {
    //     grid.evolve();
    //     float popFitness = grid.get_average_fitness();
    //     printf("Fitness of current population: %5f.5\n", popFitness);
    // }

    // CUDA_CALL(cudaUnbindTexture(fitnessTextureRef));
    // CUDA_CALL(cudaFree(fitnessTexture.device_data));

    return 0;
}
