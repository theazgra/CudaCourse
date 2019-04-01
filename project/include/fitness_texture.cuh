#pragma once
#include <cuda_utils.cuh>

struct TextureInfo
{
    byte *device_data;
    size_t memoryPitch;
    cudaChannelFormatDesc textureCFD;
};

texture<unsigned char, 2, cudaReadModeElementType> fitnessTextureRef;
TextureInfo fitnessTexture = {};