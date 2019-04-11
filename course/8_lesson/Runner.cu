#include "../../my_cuda.cu"
#include <random>
//typedef unsigned int uint;

constexpr uint ThreadPerBlock = 512;
constexpr uint BlocksPerGrid = 1024;

constexpr size_t ArraySize = 1000000;
constexpr size_t ByteArraySize = ArraySize * sizeof(int);

__global__ void find_max(int *memory, int *max)
{
	__shared__ int sMax;
	sMax = 0;

	__syncthreads();

	uint tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint strideX = blockDim.x * gridDim.x;

	while (tIdX < ArraySize)
	{
		if (memory[tIdX] > sMax)
		{
			atomicMax(&sMax, memory[tIdX]);
		}

		tIdX += strideX;
	}

	__syncthreads();
	atomicMax(max, sMax);
}

int main(int argc, char *argv[])
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> dist(-56464, 99994);

	int *host_randomArray;
	int *device_randomArray;
	host_randomArray = (int *)::operator new(ByteArraySize);

	for (size_t i = 0; i < ArraySize; i++)
	{
		host_randomArray[i] = dist(rd);
	}

	printf("Generated on cpu.\n");

	HANDLE_ERROR(cudaMalloc((void **)&device_randomArray, ByteArraySize));
	HANDLE_ERROR(cudaMemcpy(device_randomArray, host_randomArray, ByteArraySize, cudaMemcpyHostToDevice));
	free(host_randomArray);

	int *device_max;
	HANDLE_ERROR(cudaMalloc((void **)&device_max, sizeof(int)));

	CUDA_TIMED_BLOCK_START("FindMax");
	find_max<<<BlocksPerGrid, ThreadPerBlock>>>(device_randomArray, device_max);
	CUDA_TIMED_BLOCK_END;

	int max;
	HANDLE_ERROR(cudaMemcpy(&max, device_max, sizeof(int), cudaMemcpyDeviceToHost));

	printf("Found max value: %i\n", max);

	HANDLE_ERROR(cudaFree(device_max));
	HANDLE_ERROR(cudaFree(device_randomArray));

	return 0;
}
