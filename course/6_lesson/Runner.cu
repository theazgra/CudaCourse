// includes, cudaimageWidth
#include <cudaDefs.h>
#include <imageManager.h>
#include "imageKernels.cuh"

#define BLOCK_DIM 8

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

//Use the followings to store information about the input image that will be processed
unsigned char *dSrcImageData = 0;
unsigned int srcImageWidth;
unsigned int srcImageHeight;
unsigned int srcImageBPP; //Bits Per Pxel = 8, 16, 24, or 32 bit
unsigned int srcImagePitch;

//Use the followings to access the input image through the texture reference
texture<float, 2, cudaReadModeElementType> srcTexRef;
cudaChannelFormatDesc srcTexCFD;
size_t srcTexPitch;
float *dSrcTexData = 0;

size_t dstTexPitch;
uchar3 *dstTexData = 0;

KernelSetting squareKs;
float *dOutputData = 0;

template <bool normalizeTexel>
__global__ void floatHeighmapTextureToNormalmap(const unsigned int texWidth, const unsigned int texHeight, const unsigned int dstPitch, uchar3 *dst)
{
	//TODO: ...
}

#pragma region STEP 1

//TASK:	Load the input image and store loaded data in DEVICE memory (dSrcImageData)

void loadSourceImage(const char *imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	srcImageWidth = FreeImage_GetWidth(tmp);
	srcImageHeight = FreeImage_GetHeight(tmp);
	srcImageBPP = FreeImage_GetBPP(tmp);
	srcImagePitch = FreeImage_GetPitch(tmp); // FREEIMAGE aligns row data ... You have to use pitch instead of width

	cudaMalloc((void **)&dSrcImageData, srcImagePitch * srcImageHeight * srcImageBPP / 8);
	cudaMemcpy(dSrcImageData, FreeImage_GetBits(tmp), srcImagePitch * srcImageHeight * srcImageBPP / 8, cudaMemcpyHostToDevice);

	checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");
	checkDeviceMatrix<unsigned char>(dSrcImageData, srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
#pragma endregion

#pragma region STEP 2

//TASK: Create a texture based on the source image. The input images can have variable BPP (Byte Per Pixel), but finally any such image will be converted into the floating-point texture using
//		the colorToFloat kernel.

void createSrcTexure()
{
	//Floating Point Texture Data
	cudaMallocPitch((void **)&dSrcTexData, &dstTexPitch, srcImageWidth * sizeof(float), srcImageHeight);

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch (srcImageBPP)
	{
	case 8:
		colorToFloat<8><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, dstTexPitch / sizeof(float), dSrcTexData);
		break;
	case 16:
		colorToFloat<16><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, dstTexPitch / sizeof(float), dSrcTexData);
		break;
	case 24:
		colorToFloat<24><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, dstTexPitch / sizeof(float), dSrcTexData);
		break;
	case 32:
		colorToFloat<32><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, dstTexPitch / sizeof(float), dSrcTexData);
		break;
	}

	//checkDeviceMatrix<float>(dSrcTexData, texPitch, imageHeight, imageWidth, "%6.1f ", "Result of Linear Pitch Text");

	//Texture settings
	srcTexCFD = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	srcTexRef.normalized = false;
	srcTexRef.filterMode = cudaFilterModePoint;
	srcTexRef.addressMode[0] = cudaAddressModeClamp;
	srcTexRef.addressMode[1] = cudaAddressModeClamp;

	cudaBindTexture2D(0, &srcTexRef, dSrcTexData, &srcTexCFD, srcImageWidth, srcImageHeight, dstTexPitch);
}
#pragma endregion

#pragma region STEP 3

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

void createNormalMap()
{
	//TODO: Allocate Pitch memory dstTexData to store output texture
	//cudaMallocPitch(... );

	//TODO: Call the kernel that creates the normal map.
	//floatHeighmapTextureToNormalmap<true><<<squareKs.dimGrid, squareKs.dimBlock>>>( ... );

	check_data<uchar3>::checkDeviceMatrix(dstTexData, srcImageHeight, dstTexPitch / sizeof(uchar3), true, "%hhu %hhu %hhu %hhu | ", "Result of Linear Pitch Text");
}

#pragma endregion

#pragma region STEP 4

//TASK: Save output image (normal map)

void saveTexImage(const char *imageFileName)
{
	FreeImage_Initialise();

	FIBITMAP *tmp = FreeImage_Allocate(srcImageWidth, srcImageHeight, 24);
	unsigned int tmpPitch = srcImagePitch = FreeImage_GetPitch(tmp); // FREEIMAGE align row data ... You have to use pitch instead of width
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dstTexData, dstTexPitch, srcImageWidth * 3, srcImageHeight, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, imageFileName, 0);
	ImageManager::GenericWriter(tmp, imageFileName, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

#pragma endregion

void releaseMemory()
{
	cudaUnbindTexture(srcTexRef);
	if (dSrcImageData != 0)
		cudaFree(dSrcImageData);
	if (dSrcTexData != 0)
		cudaFree(dSrcTexData);
	if (dstTexData != 0)
		cudaFree(dstTexData);
	if (dOutputData)
		cudaFree(dOutputData);
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	//STEP 1
	loadSourceImage("../terrain3Kx3K.tif");

	//TODO: Setup the kernel settings
	squareKs.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	squareKs.blockSize = BLOCK_DIM * BLOCK_DIM;
	//squareKs.dimGrid = ... ;

	//Step 2 - create heighmap texture stored in the linear pitch memory
	createSrcTexure();

	//Step 3 - create the normal map
	createNormalMap();

	//Step 4 - save the normal map
	saveTexImage("d:/noramlMap.bmp");

	releaseMemory();
}
