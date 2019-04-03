#include <cell_grid.cuh>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>

////////////////////////////// Fitness texture ////////////////////////////////
struct TextureInfo
{
    byte *device_data;
    size_t pitch;
    cudaChannelFormatDesc textureCFD;
};
texture<unsigned char, 2, cudaReadModeElementType> fitnessTexRef;
TextureInfo fitnessTex = {};
///////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////// KERNELS /////////////////////////////////////////////////////////////////////

__global__ void generate_random_population(CellGridInfo gridInfo, RandomGeneratorInfo rng)
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
            int x = rng.xMin + ((int)(f1 * (rng.xMax - rng.xMin) + 0.999999));
            int y = rng.yMin + ((int)(f2 * (rng.yMax - rng.yMin) + 0.999999));

            *((Cell *)((char *)gridInfo.data + tIdY * gridInfo.pitch) + tIdX) = Cell(x, y);
            //T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;

            tIdY += strideY;
        }
        tIdX += strideX;
    }
}

// This kernel will evole current population into new one.
__global__ void evolve_kernel(const CellGridInfo currPop, CellGridInfo nextPop)
{

    uint tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint strideX = blockDim.x * gridDim.x;
    uint strideY = blockDim.y * gridDim.y;

    while (tIdX < currPop.width)
    {
        tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;
        while (tIdY < currPop.height)
        {
            Cell *cell = ((Cell *)((char *)currPop.data + tIdY * currPop.pitch) + tIdX);
            cell->fitness = tex2D<byte>(fitnessTexRef, cell->x, cell->y);

            //
            /*
            //TODO: Get cell from pitched memory.
            Cell *c = nullptr;
            //TODO: Do we allow different neighborhoods?
            Cell neighborhood[8];
            c->get_sorted_neighborhood(currentPopulation, tIdX, tIdY, neighborhood);
            Cell offspring(&neighborhood[0], &neighborhood[1]);
            offspring.random_mutation();

            //TODO: Which cell do we replace in next population? Worst in neighborhood, one parent?
            //      Replace the cell in next population.
            */
            tIdY += strideY;
        }

        tIdX += strideX;
    }
}

////////////////////////////////////////////////// END OF KERNELS /////////////////////////////////////////////////////////////////////

CellGrid::CellGrid(const size_t width, const size_t height, KernelSettings kernelSettings)
{
    this->width = width;
    this->height = height;

    this->kernelSettings = kernelSettings;
}

CellGrid::~CellGrid()
{
    // Unbind texture and release its memory.
    CUDA_CALL(cudaUnbindTexture(fitnessTexRef));
    CUDA_CALL(cudaFree(fitnessTex.device_data));

    // Release populations memory.
    if (device_currPopMemory != nullptr)
        cudaFree(device_currPopMemory);

    if (device_nextPopMemory != nullptr)
        cudaFree(device_nextPopMemory);
}

void CellGrid::create_fitness_texture(const Image &fitnessImage)
{
    assert(fitnessImage.image_type() == ImageType_GrayScale_8bpp && "Cuda texture only support 1,2 or 4 sized vectors.");

    textureWidth = fitnessImage.width();
    textureHeight = fitnessImage.height();

    size_t memoryWidth = textureWidth * fitnessImage.channel_count() * sizeof(byte);
    size_t memoryRowCount = textureHeight;

    CUDA_CALL(cudaMallocPitch((void **)&fitnessTex.device_data, &fitnessTex.pitch, memoryWidth, memoryRowCount));
    CUDA_CALL(cudaMemcpy2D(fitnessTex.device_data, fitnessTex.pitch, fitnessImage.data(), memoryWidth, memoryWidth, memoryRowCount, cudaMemcpyHostToDevice));

    fitnessTex.textureCFD = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

    fitnessTexRef.normalized = false;
    fitnessTexRef.filterMode = cudaFilterModePoint;
    fitnessTexRef.addressMode[0] = cudaAddressModeClamp;
    fitnessTexRef.addressMode[1] = cudaAddressModeClamp;

    CUDA_CALL(cudaBindTexture2D(0, &fitnessTexRef, fitnessTex.device_data, &fitnessTex.textureCFD, textureWidth, textureHeight, fitnessTex.pitch));
}

void CellGrid::initialize_grid(const Image &fitnessImage)
{
    create_fitness_texture(fitnessImage);

    // Allocate pitched memory for populations of cells.
    CUDA_CALL(cudaMallocPitch((void **)&device_currPopMemory, &currPopPitch, width * sizeof(Cell), height));
    CUDA_CALL(cudaMallocPitch((void **)&device_nextPopMemory, &nextPopPitch, width * sizeof(Cell), height));

    assert(currPopPitch == nextPopPitch && "Population memory pitch doesn't align!");

    curandStateMtgp32 *device_randomStates;
    mtgp32_kernel_params *device_kernelParams;
    size_t stateCount = rngGridDim * rngGridDim;
    assert(stateCount <= 200 && "Only 200 state params are prepared by Nvidia.");

    CUDA_CALL(cudaMalloc((void **)&device_randomStates, stateCount * sizeof(curandStateMtgp32)));
    CUDA_CALL(cudaMalloc((void **)&device_kernelParams, sizeof(mtgp32_kernel_params)));

    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, device_kernelParams));

    CURAND_CALL(curandMakeMTGP32KernelState(device_randomStates, mtgp32dc_params_fast_11213, device_kernelParams, stateCount, time(NULL)));

    CellGridInfo currPop = {};
    currPop.data = device_currPopMemory;
    currPop.pitch = currPopPitch;
    currPop.width = width;
    currPop.height = height;

    RandomGeneratorInfo rng = {};
    rng.xMin = 0;
    rng.yMin = 0;
    rng.xMax = textureWidth;
    rng.yMax = textureHeight;
    rng.state = device_randomStates;

    CUDA_TIMED_BLOCK_START("Initial population generation");
    generate_random_population<<<dim3(rngGridDim, rngGridDim, 1), dim3(rngBlockDim, rngBlockDim, 1)>>>(currPop, rng);
    CUDA_TIMED_BLOCK_END;

    CUDA_CALL(cudaFree(device_randomStates));
    // CUDA_CALL(cudaPeekAtLastError());
    // CUDA_CALL(cudaDeviceSynchronize());
    //print_cell_grid();
}

void CellGrid::print_cell_grid(bool fitness) const
{
    if (device_currPopMemory == nullptr)
        return;

    Cell *tmpMemory;
    CUDA_CALL(cudaHostAlloc((void **)&tmpMemory, currPopPitch * height, cudaHostAllocWriteCombined));
    CUDA_CALL(cudaMemcpy(tmpMemory, device_currPopMemory, currPopPitch * height, cudaMemcpyDeviceToHost));

    Cell *dataPtr = tmpMemory;
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            if (fitness)
                printf("%3i ", dataPtr[col].fitness);
            else
                printf("[%i;%i] ", dataPtr[col].x, dataPtr[col].y);
        }
        printf("\n");
        dataPtr = (Cell *)(((char *)dataPtr) + currPopPitch);
    }

    CUDA_CALL(cudaFreeHost(tmpMemory));
}

void CellGrid::evolve()
{
    CellGridInfo currPop = {};
    currPop.data = device_currPopMemory;
    currPop.pitch = currPopPitch;
    currPop.width = width;
    currPop.height = height;

    CellGridInfo nextPop = {};
    nextPop.data = device_nextPopMemory;
    nextPop.pitch = nextPopPitch;
    nextPop.width = width;
    nextPop.height = height;

    CUDA_CALL(cudaMemset2D(device_nextPopMemory, nextPopPitch, 5, width * sizeof(Cell), height));
    // Memory needs to be copied only if we decide to take some cells from old population.
    //CUDA_CALL(cudaMemcpy2D(device_nextPopMemory, nextPopPitch, device_currPopMemory, currPopPitch, width * sizeof(Cell), height, cudaMemcpyDeviceToDevice));

    CUDA_TIMED_BLOCK_START("Evolve");
    evolve_kernel<<<kernelSettings.gridDimension, kernelSettings.blockDimension>>>(currPop, nextPop);
    CUDA_TIMED_BLOCK_END;

    device_currPopMemory = device_nextPopMemory;
}

float CellGrid::get_average_fitness() const
{
    //TODO: Get fitness from current population grid by reduction.
    // float fitnessSum = 0;

    // CellGridInfo currPop = {};
    // currPop.data = device_currPopMemory;
    // currPop.memoryPitch = currPopPitch;
    // currPop.width = width;
    // currPop.height = height;

    // get_finess_kernel<<<kernelSettings.gridDimension, kernelSettings.blockDimension>>>(&currPop, &fitnessSum);

    // fitnessSum /= (width * height);
    // return fitnessSum;
}
