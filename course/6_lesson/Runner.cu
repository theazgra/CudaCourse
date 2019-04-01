// includes, cudaimageWidth
#include <cudaDefs.h>
#include <imageManager.h>
#include "imageKernels.cuh"
#include <vector>
#include "../../my_cuda.cu"

#define BLOCK_DIM 8

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

// Original image data.
unsigned char *dSrcImageData = 0;
// Original image width.
unsigned int srcImageWidth;
// Original image height.
unsigned int srcImageHeight;
// Original image bpp. (8,16,23,32, etc.)
unsigned int srcImageBPP;
// Original image pitch.
unsigned int srcImagePitch;

// Input image float texture.
texture<float, 2, cudaReadModeElementType> srcTexRef;
// Input image float textrue description.
cudaChannelFormatDesc srcTexCFD;
// Imput image float texture pitch.
size_t srcTexPitch;
// Imput image float texture data.
float *dSrcTexData = 0;

// Destination image pitch.
size_t dstTexPitch;
// Destination image data.
uchar3 *dstTexData = 0;

// Kernel settings.
KernelSetting squareKs;
// Float output data?
float *dOutputData = 0;

constexpr float zVal = 1.0f / 32.0f; //  32.0f;

template <bool normalizeTexel>
__global__ void floatHeighmapTextureToNormalmap(const unsigned int texWidth, const unsigned int texHeight,
												const unsigned int dstPitch, uchar3 *dst)
{
#if 1
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	unsigned int tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;

	while (tIdY < texHeight)
	{
		tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
		while (tIdX < texWidth)
		{
			float tl = tex2D(srcTexRef, tIdX - 1, tIdY - 1);
			float bl = tex2D(srcTexRef, tIdX - 1, tIdY + 1);
			float tr = tex2D(srcTexRef, tIdX + 1, tIdY - 1);
			float br = tex2D(srcTexRef, tIdX + 1, tIdY + 1);

			float sobelX = (-1.0f * tl) + (-2.0f * tex2D(srcTexRef, tIdX - 1, tIdY)) + (-1.0f * bl) + tr + (2.0f * tex2D(srcTexRef, tIdX + 1, tIdY)) + br;
			float sobelY = tl + (2.0f * tex2D(srcTexRef, tIdX, tIdY - 1)) + tr + (-1.0f * bl) + (-2.0f * tex2D(srcTexRef, tIdX, tIdY + 1)) + (-1.0f * br);

			tIdX += strideX;

			unsigned int dstOffset = (tIdY * dstPitch) + tIdX;
			dst[dstOffset].x = (zVal + 1) * 127.5f;
			dst[dstOffset].y = 255.0f - (sobelY + 1) * 127.5f;
			dst[dstOffset].z = (sobelX + 1) * 127.5f;
		}

		tIdY += strideY;
	}
#else
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= texWidth)
		return;
	if (y >= texHeight)
		return;
	int offset_x = gridDim.x * blockDim.x;
	int offset_y = gridDim.y * blockDim.y;

	float LeftTop;
	float LeftBot;
	float RightTop;
	float RightBot;
	float filter_y;
	float filter_x;
	uchar3 new_normal_pixel;

	while (y < texWidth)
	{

		while (x < texHeight)
		{

			//float pixely = -1 * tex2D(srcTexRef, y - 1, x - 1) + -2 * tex2D(srcTexRef, y - 1, x) + -1 * tex2D(srcTexRef, y - 1, x + 1) + 1 * tex2D(srcTexRef, y + 1, x - 1) + 2 * tex2D(srcTexRef, y + 1, x) + 1 * tex2D(srcTexRef, y + 1, x + 1);
			//float pixelx = -1 * tex2D(srcTexRef, y - 1, x - 1) + -2 * tex2D(srcTexRef, y, x - 1) + -1 * tex2D(srcTexRef, y + 1, x - 1) + 1 * tex2D(srcTexRef, y + 1, x + 1) + 2 * tex2D(srcTexRef, y , x + 1) + 1 * tex2D(srcTexRef, y + 1, x + 1);
			LeftTop = tex2D(srcTexRef, x - 1, y - 1);
			LeftBot = tex2D(srcTexRef, x + 1, y + 1);
			RightTop = tex2D(srcTexRef, x + 1, y - 1);
			RightBot = tex2D(srcTexRef, x - 1, y + 1);
			filter_y = +1 * LeftTop + +2 * tex2D(srcTexRef, x, y - 1) + +1 * RightTop - 1 * LeftBot - 2 * tex2D(srcTexRef, x, y + 1) - 1 * RightBot;
			filter_x = -1 * LeftTop + -2 * tex2D(srcTexRef, x - 1, y) + -1 * LeftBot + 1 * RightTop + 2 * tex2D(srcTexRef, x + 1, y) + 1 * RightBot;

			dst[y * dstPitch + x].z = (filter_x + 1) * 127.5f;
			dst[y * dstPitch + x].y = (filter_y + 1) * 127.5f;
			dst[y * dstPitch + x].x = (zVal + 1) * 127.5f;

			x += offset_x;
		}
		y += offset_y;
	}
#endif
}

//TASK:	Load the input image and store loaded data in DEVICE memory (dSrcImageData)
void loadSourceImage(const char *imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	srcImageWidth = FreeImage_GetWidth(tmp);
	srcImageHeight = FreeImage_GetHeight(tmp);
	srcImageBPP = FreeImage_GetBPP(tmp);
	srcImagePitch = FreeImage_GetPitch(tmp); // FREEIMAGE aligns row data ... You have to use pitch instead of width

	printf("Loaded image [%u / %u]\n", srcImageWidth, srcImageHeight);

	HANDLE_ERROR(cudaMalloc((void **)&dSrcImageData, srcImagePitch * srcImageHeight * srcImageBPP / 8));
	HANDLE_ERROR(cudaMemcpy(dSrcImageData, FreeImage_GetBits(tmp), srcImagePitch * srcImageHeight * srcImageBPP / 8, cudaMemcpyHostToDevice));

	// checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");
	// checkDeviceMatrix<unsigned char>(dSrcImageData, srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

//TASK: Create a texture based on the source image. The input images can have variable BPP (Byte Per Pixel), but finally any such image will be converted into the floating-point texture using
//		the colorToFloat kernel.

void createSrcTexure()
{
	//Floating Point Texture Data
	HANDLE_ERROR(cudaMallocPitch((void **)&dSrcTexData, &srcTexPitch,
								 srcImageWidth * sizeof(float), srcImageHeight));

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch (srcImageBPP)
	{
	case 8:
		colorToFloat<8, 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData);
		break;
	case 16:
		colorToFloat<16, 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData);
		break;
	case 24:
		colorToFloat<24, 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData);
		break;
	case 32:
		colorToFloat<32, 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData);
		break;
	}

	//Texture settings
	srcTexCFD = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	srcTexRef.normalized = false;
	srcTexRef.filterMode = cudaFilterModePoint;
	srcTexRef.addressMode[0] = cudaAddressModeClamp;
	srcTexRef.addressMode[1] = cudaAddressModeClamp;

	HANDLE_ERROR(cudaBindTexture2D(0, &srcTexRef, dSrcTexData, &srcTexCFD, srcImageWidth, srcImageHeight, srcTexPitch));
}

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

void createNormalMap()
{
	HANDLE_ERROR(cudaMallocPitch((void **)&dstTexData, &dstTexPitch, srcImageWidth * sizeof(uchar3), srcImageHeight));
	floatHeighmapTextureToNormalmap<true><<<squareKs.dimGrid, squareKs.dimBlock>>>(srcImageWidth, srcImageHeight, dstTexPitch / sizeof(uchar3), dstTexData);
}

void saveTexImage(const char *imageFileName)
{
	FreeImage_Initialise();

	FIBITMAP *tmp = FreeImage_Allocate(srcImageWidth, srcImageHeight, 24);
	unsigned int tmpPitch = srcImagePitch = FreeImage_GetPitch(tmp); // FREEIMAGE align row data ... You have to use pitch instead of width

	HANDLE_ERROR(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dstTexData, dstTexPitch, srcImageWidth * 3, srcImageHeight, cudaMemcpyDeviceToHost));

	ImageManager::GenericWriter(tmp, imageFileName, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

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

	loadSourceImage("terrain3Kx3K.tif");

	//TODO: Setup the kernel settings

	squareKs.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	squareKs.blockSize = BLOCK_DIM * BLOCK_DIM;
	squareKs.dimGrid = dim3(getNumberOfParts(srcImageWidth, BLOCK_DIM), getNumberOfParts(srcImageHeight, BLOCK_DIM));

	printf("Kernel grid: [%u / %u]; Block: [%u / %u]\n", squareKs.dimGrid.x, squareKs.dimGrid.y, squareKs.dimBlock.x, squareKs.dimBlock.y);

	//Step 2 - create heighmap texture stored in the linear pitch memory
	createSrcTexure();

	//Step 3 - create the normal map
	createNormalMap();

	//Step 4 - save the normal map
	saveTexImage("normal.png");

	releaseMemory();
}
