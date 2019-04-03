#include <cell_grid.cuh>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>

// Number of random status must be <= 200, because only 200 state params are prepared by nvidia, more params are definitely possible
// but the user must generate them. 14*14*1*1 kernel will use 196 states which is as much as possible.
constexpr uint rngGridDim = 14;
constexpr uint rngBlockDim = 1;

struct RandomGeneratorInfo
{
    curandStateMtgp32 *state;
    int xMin;
    int yMin;
    int xMax;
    int yMax;
};

__global__ static void generate_random_population(CellGridInfo gridInfo, RandomGeneratorInfo rng)
{
    uint tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint strideX = blockDim.x * gridDim.x;
    uint strideY = blockDim.y * gridDim.y;

    uint rngStateOffset = (tIdX * rngGridDim) + tIdY;

    while (tIdX < gridInfo.width)
    {
        tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;
        while (tIdY < gridInfo.height)
        {
            float f1 = curand_uniform(&rng.state[rngStateOffset]);
            float f2 = curand_uniform(&rng.state[rngStateOffset]);
            int x = (int)(f1 * (rng.xMax - rng.xMin) + 0.999999);
            int y = (int)(f2 * (rng.yMax - rng.yMin) + 0.999999);

            *((Cell *)((char *)gridInfo.data + tIdY * gridInfo.pitch) + tIdX) = Cell(x, y);
            //T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;

            tIdY += strideY;
        }
        tIdX += strideX;
    }
}
CellGrid::CellGrid(const size_t width, const size_t height, KernelSettings kernelSettings)
{
    this->width = width;
    this->height = height;

    this->kernelSettings = kernelSettings;
}

CellGrid::~CellGrid()
{
    if (device_currPopMemory != nullptr)
    {
        cudaFree(device_currPopMemory);
    }

    if (device_nextPopMemory != nullptr)
    {
        cudaFree(device_nextPopMemory);
    }
}

void CellGrid::initialize_grid()
{
    // Allocate pitched memory for populations of cells.
    CUDA_CALL(cudaMallocPitch((void **)&device_currPopMemory, &currPopMemoryPitch, width * sizeof(Cell), height));
    CUDA_CALL(cudaMallocPitch((void **)&device_nextPopMemory, &nextPopMemoryPitch, width * sizeof(Cell), height));

    assert(currPopMemoryPitch == nextPopMemoryPitch && "Population memory pitch doesn't align!");

    curandStateMtgp32 *device_randomStates;
    mtgp32_kernel_params *device_kernelParams;
    size_t stateCount = rngGridDim * rngGridDim;
    assert(stateCount <= 200 && "Only 200 state params are prepared by Nvidia.");

    CUDA_CALL(cudaMalloc((void **)&device_randomStates, stateCount * sizeof(curandStateMtgp32)));
    CUDA_CALL(cudaMalloc((void **)&device_kernelParams, sizeof(mtgp32_kernel_params)));

    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, device_kernelParams));
    CURAND_CALL(curandMakeMTGP32KernelState(device_randomStates, mtgp32dc_params_fast_11213, device_kernelParams, stateCount, 50000));

    CellGridInfo currPop = {};
    currPop.data = device_currPopMemory;
    currPop.pitch = currPopMemoryPitch;
    currPop.width = width;
    currPop.height = height;

    RandomGeneratorInfo rng = {};
    rng.xMin = 0;
    rng.yMin = 0;
    rng.xMax = 1024;
    rng.yMax = 1024;
    rng.state = device_randomStates;

    CUDA_TIMED_BLOCK_START("Initial population generation");
    generate_random_population<<<dim3(rngGridDim, rngGridDim, 1), dim3(rngBlockDim, rngBlockDim, 1)>>>(currPop, rng);
    CUDA_TIMED_BLOCK_END;

    CUDA_CALL(cudaFree(device_randomStates));
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    //print_cell_grid();
}

void CellGrid::print_cell_grid() const
{
    if (device_currPopMemory == nullptr)
        return;

    Cell *tmpMemory;
    CUDA_CALL(cudaHostAlloc((void **)&tmpMemory, currPopMemoryPitch * height, cudaHostAllocWriteCombined));
    CUDA_CALL(cudaMemcpy(tmpMemory, device_currPopMemory, currPopMemoryPitch * height, cudaMemcpyDeviceToHost));

    Cell *dataPtr = tmpMemory;
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            printf("[%i;%i] ", dataPtr[col].x, dataPtr[col].y);
        }
        printf("\n");
        dataPtr = (Cell *)(((char *)dataPtr) + currPopMemoryPitch);
    }

    CUDA_CALL(cudaFreeHost(tmpMemory));
}

void CellGrid::evolve()
{
    // CellGridInfo currPop = {};
    // currPop.data = device_currPopMemory;
    // currPop.memoryPitch = currPopMemoryPitch;
    // currPop.width = width;
    // currPop.height = height;

    // CellGridInfo nextPop = {};
    // nextPop.data = device_nextPopMemory;
    // nextPop.memoryPitch = nextPopMemoryPitch;
    // nextPop.width = width;
    // nextPop.height = height;

    // // This will work only if memory pitch are same?
    // CUDA_CALL(cudaMemcpy2D(device_nextPopMemory, nextPopMemoryPitch, device_currPopMemory,
    //                           currPopMemoryPitch, width * sizeof(Cell), height, cudaMemcpyDeviceToDevice));

    // evolve_kernel<<<kernelSettings.gridDimension, kernelSettings.blockDimension>>>(&currPop, &nextPop);

    // // Swap populations.
    // device_currPopMemory = device_nextPopMemory;
}

float CellGrid::get_average_fitness() const
{
    //TODO: Get fitness from current population grid by reduction.
    // float fitnessSum = 0;

    // CellGridInfo currPop = {};
    // currPop.data = device_currPopMemory;
    // currPop.memoryPitch = currPopMemoryPitch;
    // currPop.width = width;
    // currPop.height = height;

    // get_finess_kernel<<<kernelSettings.gridDimension, kernelSettings.blockDimension>>>(&currPop, &fitnessSum);

    // fitnessSum /= (width * height);
    // return fitnessSum;
}