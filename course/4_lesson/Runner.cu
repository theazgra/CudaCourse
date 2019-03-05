#include "../../my_cuda.cu"

template <class T>
__host__ void checkDeviceMatrix(const T *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const char *format = "%f ", const char *message = "")
{
	printf("\nDEVICE MEMORY: %s [%u %u]\n", message, rows, cols);
	T *ptr;
	cudaHostAlloc((void **)&ptr, pitchInBytes * rows, cudaHostAllocWriteCombined);
	cudaMemcpy(ptr, m, rows * pitchInBytes, cudaMemcpyDeviceToHost);
	T *p = ptr;
	for (unsigned int i = 0; i < rows; i++)
	{
		for (unsigned int j = 0; j < cols; j++)
		{
			printf(format, p[j]);
		}
		printf("\n");
		p = (T *)(((char *)p) + pitchInBytes);
	}
	cudaFreeHost(ptr);
}

//WARNING!!! Do not change TPB and NO_FORCES for this demo !!!
constexpr unsigned int TPB = 128;
constexpr unsigned int NO_FORCES = 256;
constexpr unsigned int NO_RAIN_DROPS = 1 << 20;

constexpr unsigned int MEM_BLOCKS_PER_THREAD_BLOCK = 8;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

float3 *createData(const unsigned int length)
{
	std::random_device rd;
	std::mt19937_64 mt(rd());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	// ::operator new == malloc
	float3 *data = static_cast<float3 *>(::operator new(sizeof(float3) * length));

	for (size_t i = 0; i < length; i++)
	{
		// data[i] = make_float3(dist(rd), dist(rd), dist(rd));
		data[i] = make_float3(1.0f, 1.0f, 1.0f);
	}

	return data;
}

void printData(const float3 *data, const unsigned int length)
{
	if (data == 0)
		return;
	const float3 *ptr = data;
	for (unsigned int i = 0; i < length; i++, ptr++)
	{
		printf("%5.2f %5.2f %5.2f ", ptr->x, ptr->y, ptr->z);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Sums the forces to get the final one using parallel reduction.
/// 		    WARNING!!! The method was written to meet input requirements of our example, i.e. 128 threads and 256 forces  </summary>
/// <param name="dForces">	  	The forces. </param>
/// <param name="noForces">   	The number of forces. </param>
/// <param name="dFinalForce">	[in,out] If non-null, the final force. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void reduce(const float3 *__restrict__ dForces, const unsigned int noForces, float3 *__restrict__ dFinalForce)
{
	__shared__ float3 sForces[TPB]; //SEE THE WARNING MESSAGE !!!
	unsigned int tid = threadIdx.x;
	unsigned int next = TPB; //SEE THE WARNING MESSAGE !!!

	float3 *src = &sForces[tid];
	float3 *src2 = (float3 *)&dForces[tid + next];

	*src = dForces[tid];

	src->x += src2->x;
	src->y += src2->y;
	src->z += src2->z;

	__syncthreads();

	//next/2
	next >>= 1; //=64
	if (tid >= next)
		return;

	src2 = src + next;

	src->x += src2->x;
	src->y += src2->y;
	src->z += src2->z;

	__syncthreads();

	next >>= 1; //=32
	if (tid >= next)
		return;

	volatile float3 *vsrc = &sForces[tid];
	volatile float3 *vsrc2 = vsrc + next;

	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; //=16
	if (tid >= next)
		return;

	vsrc2 = vsrc + next;

	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; //=8
	if (tid >= next)
		return;

	vsrc2 = vsrc + next;

	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; //=4
	if (tid >= next)
		return;

	vsrc2 = vsrc + next;

	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; //=2
	if (tid >= next)
		return;

	vsrc2 = vsrc + next;

	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; //=1
	if (tid >= next)
		return;

	vsrc2 = vsrc + next;

	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	dFinalForce->x = vsrc->x;
	dFinalForce->y = vsrc->y;
	dFinalForce->z = vsrc->z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds the FinalForce to every Rain drops position. </summary>
/// <param name="dFinalForce">	The final force. </param>
/// <param name="noRainDrops">	The number of rain drops. </param>
/// <param name="dRainDrops"> 	[in,out] If non-null, the rain drops positions. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void add(const float3 *__restrict__ dFinalForce, const unsigned int noRainDrops, float3 *__restrict__ dRainDrops)
{
	//TODO: Add the FinalForce to every Rain drops position.
	uint xOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint xSkip = gridDim.x * blockDim.x;

	while (xOffset < noRainDrops)
	{
		dRainDrops[xOffset].x += dFinalForce->x;
		dRainDrops[xOffset].y += dFinalForce->y;
		dRainDrops[xOffset].z += dFinalForce->z;

		xOffset += xSkip;
	}
}

int main(int argc, char *argv[])
{
	CUDA_TIMED_BLOCK_START("Drop simulation.");

	float3 *hForces = createData(NO_FORCES);
	float3 *hDrops = createData(NO_RAIN_DROPS);

	float3 *dForces = nullptr;
	float3 *dDrops = nullptr;
	float3 *dFinalForce = nullptr;

	error = cudaMalloc((void **)&dForces, NO_FORCES * sizeof(float3));
	error = cudaMemcpy(dForces, hForces, NO_FORCES * sizeof(float3), cudaMemcpyHostToDevice);

	error = cudaMalloc((void **)&dDrops, NO_RAIN_DROPS * sizeof(float3));
	error = cudaMemcpy(dDrops, hDrops, NO_RAIN_DROPS * sizeof(float3), cudaMemcpyHostToDevice);

	error = cudaMalloc((void **)&dFinalForce, sizeof(float3));

	KernelSetting ksReduce;
	ksReduce.dimBlock = dim3(TPB, 1);
	ksReduce.dimGrid = dim3(1, 1);

	KernelSetting ksAdd;
	ksAdd.dimBlock = dim3(TPB, 1);
	ksAdd.dimGrid = dim3(get_number_of_parts(NO_RAIN_DROPS, TPB), 1);

	for (unsigned int i = 0; i < 1000; i++)
	{
		reduce<<<ksReduce.dimGrid, ksReduce.dimBlock>>>(dForces, NO_FORCES, dFinalForce);
		add<<<ksAdd.dimGrid, ksAdd.dimBlock>>>(dFinalForce, NO_RAIN_DROPS, dDrops);
	}

	checkDeviceMatrix<float>((float *)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
	checkDeviceMatrix<float>((float *)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

	if (hForces)
		free(hForces);
	if (hDrops)
		free(hDrops);

	cudaFree(dForces);
	cudaFree(dDrops);

	CUDA_TIMED_BLOCK_END;
}
