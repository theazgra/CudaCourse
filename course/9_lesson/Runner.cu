#include "../../my_cuda.cu"
#include <time.h>
#include <math.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

typedef unsigned int uint;

constexpr uint N = 1 << 20;
constexpr uint MEMSIZE = N * sizeof(uint);
constexpr uint NO_LOOPS = 100;
constexpr uint THREAD_PER_BLOCK = 256;
constexpr uint GRID_SIZE = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

void fillData(uint *data, const uint length)
{
	//srand(time(0));
	for (uint i = 0; i < length; i++)
	{
		//data[i]= rand();
		data[i] = 1;
	}
}

void printData(const uint *data, const uint length)
{
	if (data == 0)
		return;
	for (uint i = 0; i < length; i++)
	{
		printf("%u ", data[i]);
	}
	printf("\n");
}

__global__ void kernel(const uint *__restrict__ a, const uint *__restrict__ b, const uint length, uint *c)
{
	uint tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const uint stride = blockDim.x * gridDim.x;
	while (tid < length)
	{
		c[tid] = a[tid] + b[tid];
		tid += stride;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test1()
{
	uint *a, *b, *c;
	uint *da, *db, *dc;

	// paged-locked allocation
	// CUDA_CALL(cudaHostAlloc((void **)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	// CUDA_CALL(cudaHostAlloc((void **)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	// CUDA_CALL(cudaHostAlloc((void **)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));

	CUDA_CALL(cudaMallocHost((void **)&a, NO_LOOPS * MEMSIZE));
	CUDA_CALL(cudaMallocHost((void **)&b, NO_LOOPS * MEMSIZE));
	CUDA_CALL(cudaMallocHost((void **)&c, NO_LOOPS * MEMSIZE));

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	CUDA_CALL(cudaMalloc((void **)&da, MEMSIZE));
	CUDA_CALL(cudaMalloc((void **)&db, MEMSIZE));
	CUDA_CALL(cudaMalloc((void **)&dc, MEMSIZE));

	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(&stream));

	uint dataOffset = 0;

	CUDA_TIMED_BLOCK_START("Single stream");

	for (int i = 0; i < NO_LOOPS; i++)
	{
		//TODO:  copy a->da, b->db
		CUDA_CALL(cudaMemcpyAsync(da, a + dataOffset, MEMSIZE, cudaMemcpyHostToDevice, stream));
		CUDA_CALL(cudaMemcpyAsync(db, b + dataOffset, MEMSIZE, cudaMemcpyHostToDevice, stream));
		//TODO:  run the kernel in the stream
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream>>>(da, db, N, dc);
		//TODO:  copy dc->c
		CUDA_CALL(cudaMemcpyAsync(c + dataOffset, dc, MEMSIZE, cudaMemcpyDeviceToHost, stream));
		dataOffset += N;
	}

	//TODO: Synchonize stream
	CUDA_CALL(cudaStreamSynchronize(stream));
	//TODO: Destroy stream
	CUDA_CALL(cudaStreamDestroy(stream));

	CUDA_TIMED_BLOCK_END;

	printData(c, 100);

	CUDA_CALL(cudaFree(da));
	CUDA_CALL(cudaFree(db));
	CUDA_CALL(cudaFree(dc));

	CUDA_CALL(cudaFreeHost(a));
	CUDA_CALL(cudaFreeHost(b));
	CUDA_CALL(cudaFreeHost(c));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test2()
{
	uint *a, *b, *c;
	uint *da, *db, *dc;

	CUDA_CALL(cudaMallocHost((void **)&a, NO_LOOPS * MEMSIZE));
	CUDA_CALL(cudaMallocHost((void **)&b, NO_LOOPS * MEMSIZE));
	CUDA_CALL(cudaMallocHost((void **)&c, NO_LOOPS * MEMSIZE));

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	CUDA_CALL(cudaMalloc((void **)&da, MEMSIZE));
	CUDA_CALL(cudaMalloc((void **)&db, MEMSIZE));
	CUDA_CALL(cudaMalloc((void **)&dc, MEMSIZE));

	cudaStream_t stream_1, stream_2;
	CUDA_CALL(cudaStreamCreate(&stream_1));
	CUDA_CALL(cudaStreamCreate(&stream_2));

	uint dataOffset = 0;

	CUDA_TIMED_BLOCK_START("2 Streams - Depth first");

	for (int i = 0; i < NO_LOOPS; i += 2)
	{
		uint offset_1 = dataOffset;
		uint offset_2 = dataOffset + N;

		// Stream 1
		CUDA_CALL(cudaMemcpyAsync(da, a + offset_1, MEMSIZE, cudaMemcpyHostToDevice, stream_1));
		CUDA_CALL(cudaMemcpyAsync(db, b + offset_1, MEMSIZE, cudaMemcpyHostToDevice, stream_1));
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream_1>>>(da, db, N, dc);
		CUDA_CALL(cudaMemcpyAsync(c + offset_1, dc, MEMSIZE, cudaMemcpyDeviceToHost, stream_1));

		// Stream 2
		CUDA_CALL(cudaMemcpyAsync(da, a + offset_2, MEMSIZE, cudaMemcpyHostToDevice, stream_2));
		CUDA_CALL(cudaMemcpyAsync(db, b + offset_2, MEMSIZE, cudaMemcpyHostToDevice, stream_2));
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream_2>>>(da, db, N, dc);
		CUDA_CALL(cudaMemcpyAsync(c + offset_2, dc, MEMSIZE, cudaMemcpyDeviceToHost, stream_2));

		dataOffset += 2 * N;
	}

	//TODO: Synchonize stream
	CUDA_CALL(cudaStreamSynchronize(stream_1));
	CUDA_CALL(cudaStreamSynchronize(stream_2));
	//TODO: Destroy stream
	CUDA_CALL(cudaStreamDestroy(stream_1));
	CUDA_CALL(cudaStreamDestroy(stream_2));

	CUDA_TIMED_BLOCK_END;

	printData(c, 100);

	CUDA_CALL(cudaFree(da));
	CUDA_CALL(cudaFree(db));
	CUDA_CALL(cudaFree(dc));

	CUDA_CALL(cudaFreeHost(a));
	CUDA_CALL(cudaFreeHost(b));
	CUDA_CALL(cudaFreeHost(c));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test3()
{
	uint *a, *b, *c;
	uint *da, *db, *dc;

	CUDA_CALL(cudaMallocHost((void **)&a, NO_LOOPS * MEMSIZE));
	CUDA_CALL(cudaMallocHost((void **)&b, NO_LOOPS * MEMSIZE));
	CUDA_CALL(cudaMallocHost((void **)&c, NO_LOOPS * MEMSIZE));

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	CUDA_CALL(cudaMalloc((void **)&da, MEMSIZE));
	CUDA_CALL(cudaMalloc((void **)&db, MEMSIZE));
	CUDA_CALL(cudaMalloc((void **)&dc, MEMSIZE));

	cudaStream_t stream_1, stream_2;
	CUDA_CALL(cudaStreamCreate(&stream_1));
	CUDA_CALL(cudaStreamCreate(&stream_2));

	uint dataOffset = 0;

	CUDA_TIMED_BLOCK_START("2 Streams - Breadth first");

	for (int i = 0; i < NO_LOOPS; i += 2)
	{
		uint offset_1 = dataOffset;
		uint offset_2 = dataOffset + N;

		CUDA_CALL(cudaMemcpyAsync(da, a + offset_1, MEMSIZE, cudaMemcpyHostToDevice, stream_1));
		CUDA_CALL(cudaMemcpyAsync(da, a + offset_2, MEMSIZE, cudaMemcpyHostToDevice, stream_2));

		CUDA_CALL(cudaMemcpyAsync(db, b + offset_1, MEMSIZE, cudaMemcpyHostToDevice, stream_1));
		CUDA_CALL(cudaMemcpyAsync(db, b + offset_2, MEMSIZE, cudaMemcpyHostToDevice, stream_2));

		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream_1>>>(da, db, N, dc);
		kernel<<<GRID_SIZE, THREAD_PER_BLOCK, 0, stream_2>>>(da, db, N, dc);

		CUDA_CALL(cudaMemcpyAsync(c + offset_1, dc, MEMSIZE, cudaMemcpyDeviceToHost, stream_1));
		CUDA_CALL(cudaMemcpyAsync(c + offset_2, dc, MEMSIZE, cudaMemcpyDeviceToHost, stream_2));

		dataOffset += 2 * N;
	}

	//TODO: Synchonize stream
	CUDA_CALL(cudaStreamSynchronize(stream_1));
	CUDA_CALL(cudaStreamSynchronize(stream_2));
	//TODO: Destroy stream
	CUDA_CALL(cudaStreamDestroy(stream_1));
	CUDA_CALL(cudaStreamDestroy(stream_2));

	CUDA_TIMED_BLOCK_END;

	printData(c, 100);

	CUDA_CALL(cudaFree(da));
	CUDA_CALL(cudaFree(db));
	CUDA_CALL(cudaFree(dc));

	CUDA_CALL(cudaFreeHost(a));
	CUDA_CALL(cudaFreeHost(b));
	CUDA_CALL(cudaFreeHost(c));
}

int main(int argc, char *argv[])
{
	test1();
	test2();
	test3();

	return 0;
}
