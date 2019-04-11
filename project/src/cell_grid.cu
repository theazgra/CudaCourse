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

            Cell rnd(x, y);
            //rnd.fitness = tex2D<byte>(fitnessTexRef, x, y);
            rnd.fitness = 1.0f;
            // *((Cell *)((char *)gridInfo.data + tIdY * gridInfo.pitch) + tIdX) = rnd; // Cell(x, y);
            // T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;

            gridInfo.data[(tIdY * gridInfo.width) + tIdX] = rnd;

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
            //Cell *cell = ((Cell *)((char *)currPop.data + tIdY * currPop.pitch) + tIdX);
            Cell *cell = &currPop.data[(tIdY * currPop.width) + tIdX];

            // Initial fitness value is set in initialization code.
            //cell->fitness = (float)tex2D<byte>(fitnessTexRef, cell->x, cell->y);

            // We can't find partner in cell code, becuse we don't know the fitness value.
            // We would have to do 2 iteratios of this loops. One beforehand to just setup fitness value,
            // then synchronize all threads and find the mating partner.

            Cell *partner = nullptr;
            float bestFitness = -1;
            {
                // L5 cross
                // Cell *topCell = ((Cell *)((char *)currPop.data + mod(tIdY - 1, currPop.height) * currPop.pitch) + tIdX);
                // Cell *bottomCell = ((Cell *)((char *)currPop.data + mod(tIdY + 1, currPop.height) * currPop.pitch) + tIdX);
                // Cell *leftCell = ((Cell *)((char *)currPop.data + tIdY * currPop.pitch) + mod(tIdX - 1, currPop.width));
                // Cell *rightCell = ((Cell *)((char *)currPop.data + tIdY * currPop.pitch) + mod(tIdX + 1, currPop.width));

                Cell *topCell = &currPop.data[(mod(tIdY - 1, currPop.height) * currPop.width) + tIdX];
                Cell *bottomCell = &currPop.data[(mod(tIdY + 1, currPop.height) * currPop.width) + tIdX];
                Cell *leftCell = &currPop.data[(tIdY * currPop.width) + mod(tIdX - 1, currPop.width)];
                Cell *rightCell = &currPop.data[(tIdY * currPop.width) + mod(tIdX + 1, currPop.width)];

                float topCellFitness = (float)tex2D<byte>(fitnessTexRef, topCell->x, topCell->y);
                float bottomCellFitness = (float)tex2D<byte>(fitnessTexRef, bottomCell->x, bottomCell->y);
                float leftCellFitness = (float)tex2D<byte>(fitnessTexRef, leftCell->x, leftCell->y);
                float rightCellFitness = (float)tex2D<byte>(fitnessTexRef, rightCell->x, rightCell->y);

                partner = topCell;
                bestFitness = topCellFitness;

                if (bottomCellFitness > bestFitness)
                {
                    bestFitness = bottomCellFitness;
                    partner = bottomCell;
                }
                if (leftCellFitness > bestFitness)
                {
                    bestFitness = leftCellFitness;
                    partner = leftCell;
                }
                if (rightCellFitness > bestFitness)
                {
                    bestFitness = rightCellFitness;
                    partner = rightCell;
                }
            }

            Cell offspring = Cell(cell, partner);
            // offspring.random_mutation();
            offspring.fitness = (float)tex2D<byte>(fitnessTexRef, offspring.x, offspring.y);
            //offspring.fitness = 1.0f;
            //*((Cell *)((char *)nextPop.data + tIdY * nextPop.pitch) + tIdX) = offspring;
            nextPop.data[(tIdY * nextPop.width) + tIdX] = offspring;

            tIdY += strideY;
        }
        tIdX += strideX;
    }
}

__global__ void stupid_reduce(CellGridInfo grid, float *colSum)
{
    uint tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint strideX = blockDim.x * gridDim.x;

    while (tIdX < grid.width)
    {
        //Calculate the sum of the column;
        //colSum[tIdX] = 1.0f;

        for (uint row = 0; row < grid.height; row++)
        {
            //Cell *cell = ((Cell *)((char *)grid.data + row * grid.pitch) + tIdX);
            Cell *cell = &grid.data[(row * grid.width) + tIdX];
            colSum[tIdX] += cell->fitness;
        }

        tIdX += strideX;
    }
}

template <unsigned int blockSize>
__device__ void warp_reduce(volatile float *sData, unsigned int tId)
{
    if (blockSize >= 64)
        sData[tId] += sData[tId + 32];
    if (blockSize >= 32)
        sData[tId] += sData[tId + 16];
    if (blockSize >= 16)
        sData[tId] += sData[tId + 8];
    if (blockSize >= 8)
        sData[tId] += sData[tId + 4];
    if (blockSize >= 4)
        sData[tId] += sData[tId + 2];
    if (blockSize >= 2)
        sData[tId] += sData[tId + 1];
}

template <unsigned int blockSize>
__global__ void smart_reduce(CellGridInfo grid, float *sums, unsigned int n)
{
    extern __shared__ float sData[];
    unsigned int tId = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tId;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sData[tId] = 0.0f;

    while (i < n)
    {
        sData[tId] += grid.data[i].fitness + grid.data[i + blockSize].fitness;
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512)
    {
        if (tId < 256)
        {
            sData[tId] += sData[tId + 256];
        }
        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tId < 128)
        {
            sData[tId] += sData[tId + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tId < 64)
        {
            sData[tId] += sData[tId + 64];
        }
        __syncthreads();
    }

    if (tId < 32)
        warp_reduce<blockSize>(sData, tId);

    if (tId == 0)
        sums[blockIdx.x] = sData[0];
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
    // CUDA_CALL(cudaMallocPitch((void **)&device_currPopMemory, &currPopPitch, width * sizeof(Cell), height));
    // CUDA_CALL(cudaMallocPitch((void **)&device_nextPopMemory, &nextPopPitch, width * sizeof(Cell), height));

    //NOTE: For now we are using normal un-pitched memory.
    currPopPitch = nextPopPitch = width * sizeof(Cell);
    CUDA_CALL(cudaMalloc((void **)&device_currPopMemory, width * height * sizeof(Cell)));
    CUDA_CALL(cudaMalloc((void **)&device_nextPopMemory, width * height * sizeof(Cell)));

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
    printf("RNG interval xMax: %u yMax: %u\n", rng.xMax, rng.yMax);
    rng.state = device_randomStates;

    CUDA_TIMED_BLOCK_START("Initial population generation");
    generate_random_population<<<dim3(rngGridDim, rngGridDim, 1), dim3(rngBlockDim, rngBlockDim, 1)>>>(currPop, rng);
    CUDA_TIMED_BLOCK_END;

    CUDA_CALL(cudaFree(device_randomStates));
    // CUDA_CALL(cudaPeekAtLastError());
    // CUDA_CALL(cudaDeviceSynchronize());
    //print_cell_grid(device_currPopMemory, currPopPitch, true);
    printf("Grid initialized\n");
}

void CellGrid::print_cell_grid(const Cell *data, const size_t pitch, bool fitness) const
{
    if (device_currPopMemory == nullptr)
        return;

    Cell *tmpMemory;
    CUDA_CALL(cudaHostAlloc((void **)&tmpMemory, pitch * height, cudaHostAllocWriteCombined));
    CUDA_CALL(cudaMemcpy(tmpMemory, data, pitch * height, cudaMemcpyDeviceToHost));

    Cell *dataPtr = tmpMemory;
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            if (fitness)
                printf("%2.1f ", dataPtr[col].fitness);
            else
                printf("[%i;%i] ", dataPtr[col].x, dataPtr[col].y);
        }
        printf("\n");
        dataPtr = (Cell *)(((char *)dataPtr) + pitch);
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

    //print_cell_grid(device_currPopMemory, currPopPitch, true);
    //printf("--------------------------------------------------------------------------------\n");
    //print_cell_grid(device_nextPopMemory, nextPopPitch, true);
    //device_currPopMemory = device_nextPopMemory;

    //CUDA_CALL(cudaMemcpy(device_currPopMemory, device_nextPopMemory, sizeof(Cell)*width*height, cudaMemcpyDeviceToDevice));

    Cell *tmp = device_currPopMemory;
    device_currPopMemory = device_nextPopMemory;
    device_nextPopMemory = tmp;
}

float CellGrid::get_average_fitness() const
{
    unsigned int n = width * height;
    constexpr unsigned int ReduceTPB = 512;
    unsigned int numberOfBlocks = get_number_of_parts(n, ReduceTPB);
    printf("number of blocks %u\n", numberOfBlocks);

    dim3 dimGrid = dim3(numberOfBlocks, 1, 1);
    dim3 dimBlock = dim3(ReduceTPB, 1, 1);
    unsigned int sMemSize = ReduceTPB * sizeof(float);

    CellGridInfo gridInfo = {};
    gridInfo.data = device_currPopMemory;
    gridInfo.pitch = currPopPitch;
    gridInfo.width = width;
    gridInfo.height = height;

    CUDA_TIMED_BLOCK_START("complete_smart_reduce");
    float *device_sums;
    CUDA_CALL(cudaMalloc((void **)&device_sums, numberOfBlocks * sizeof(float)));

    smart_reduce<ReduceTPB><<<dimGrid, dimBlock, sMemSize>>>(gridInfo, device_sums, n);

    float *hostSum = (float *)::operator new(numberOfBlocks * sizeof(float));
    CUDA_CALL(cudaMemcpy(hostSum, device_sums, numberOfBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0;
    for (size_t i = 0; i < numberOfBlocks; i++)
    {
        sum += hostSum[i];
    }

    CUDA_CALL(cudaFree(device_sums));
    free(hostSum);

    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_TIMED_BLOCK_END;
    return sum / (float)n;

    /*
    float *device_colSums;
    CUDA_CALL(cudaMalloc((void **)&device_colSums, sizeof(float) * width));
    CUDA_CALL(cudaMemset(device_colSums, 0, sizeof(float) * width));

    CellGridInfo currPop = {};
    currPop.data = device_currPopMemory;
    currPop.pitch = currPopPitch;
    currPop.width = width;
    currPop.height = height;

    dim3 bd = dim3(32, 1, 1);
    dim3 gd = dim3(get_number_of_parts(width, 32), 1, 1);
    CUDA_TIMED_BLOCK_START("stupid_reduce");
    stupid_reduce<<<gd, bd>>>(currPop, device_colSums);
    CUDA_TIMED_BLOCK_END;

    float sum = 0.0f;
    float *colSums;

    colSums = (float *)::operator new(sizeof(float) * width);
    CUDA_CALL(cudaMemcpy(colSums, device_colSums, sizeof(float) * width, cudaMemcpyDeviceToHost));
    for (uint col = 0; col < width; col++)
        sum += colSums[col];

    CUDA_CALL(cudaFree(device_colSums));
    free(colSums);

    return (sum / (float)(width * height));
    */
}
