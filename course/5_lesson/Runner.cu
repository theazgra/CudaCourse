// #include "../../my_cuda.cu"

#include <cudaDefs.h>
#include <imageManager.h>
#include "imageKernels.cuh"

#define BLOCK_DIM 8

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

texture<float, 2, cudaReadModeElementType> texRef; // declared texture reference must be at file-scope !!!

cudaChannelFormatDesc texChannelDesc;

unsigned char *dImageData = 0;
unsigned int imageWidth;
unsigned int imageHeight;
unsigned int imageBPP; //Bits Per Pixel = 8, 16, 24, or 32 bit
unsigned int imagePitch;

size_t texPitch;
float *dLinearPitchTextureData = 0;
cudaArray *dArrayTextureData = 0;

KernelSetting ks;

float *dOutputData = 0;

void loadSourceImage(const char *imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp); // FREEIMAGE align row data ... You have to use pitch instead of width

	cudaMalloc((void **)&dImageData, imagePitch * imageHeight * imageBPP / 8);
	cudaMemcpy(dImageData, FreeImage_GetBits(tmp), imagePitch * imageHeight * imageBPP / 8, cudaMemcpyHostToDevice);

	checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), imagePitch, imageHeight, imageWidth, "%hhu ", "Result of Linear Pitch Text");
	checkDeviceMatrix<unsigned char>(dImageData, imagePitch, imageHeight, imageWidth, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

void createTextureFromLinearPitchMemory()
{
	cudaMallocPitch((void **)&dLinearPitchTextureData, &texPitch, imageWidth * sizeof(float), imageHeight);

	switch (imageBPP)
	{
	case 8:
		colorToFloat<8><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	case 16:
		colorToFloat<16><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	case 24:
		colorToFloat<24><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	case 32:
		colorToFloat<32><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	default:
		break;
	}

	checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "%6.1f ", "Result of Linear Pitch Text");

	texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	texRef.normalized = false;
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;
	texRef.filterMode = cudaFilterModePoint;

	cudaBindTexture2D(0, &texRef, dLinearPitchTextureData, &texChannelDesc, imageWidth, imageHeight, texPitch);
}

void createTextureFrom2DArray()
{
	//TODO: Define texture (texRef) parameters

	//TODO: Define texture channel descriptor (texChannelDesc)
	//texChannelDesc = ...

	//Converts custom image data to float and stores result in the float_linear_data
	float *dLinearTextureData = 0;
	cudaMalloc((void **)&dLinearTextureData, imageWidth * imageHeight * sizeof(float));
	switch (imageBPP)
	{
		//TODO: Here call your kernel to convert image into linear memory (no pitch!!!)
	}

	cudaMallocArray(&dArrayTextureData, &texChannelDesc, imageWidth, imageHeight);

	//TODO: copy data into cuda array (dArrayTextureData)
	//cudaMemcpyToArray(...);

	//TODO: Bind texture
	//cudaBind...

	cudaFree(dLinearTextureData);
}

void releaseMemory()
{
	cudaUnbindTexture(texRef);
	if (dImageData != 0)
		cudaFree(dImageData);
	if (dLinearPitchTextureData != 0)
		cudaFree(dLinearPitchTextureData);
	if (dArrayTextureData)
		cudaFreeArray(dArrayTextureData);
	if (dOutputData)
		cudaFree(dOutputData);
}

__global__ void texKernel(const unsigned int texWidth, const unsigned int texHeight, float *dst)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//TODO some kernel
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	loadSourceImage("terrain10x10.tif");

	cudaMalloc((void **)&dOutputData, imageWidth * imageHeight * sizeof(float));

	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimGrid = dim3((imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	//Test 1 - texture stored in linear pitch memory
	createTextureFromLinearPitchMemory();
	texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
	checkDeviceMatrix<float>(dOutputData, imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

	//Test 2 - texture stored in 2D array
	createTextureFrom2DArray();
	texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
	checkDeviceMatrix<float>(dOutputData, imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

	releaseMemory();
}
