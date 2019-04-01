#include <cell_grid.cuh>

CellGrid::CellGrid(const size_t width, const size_t height, KernelSettings kernelSettings)
{
    this->width = width;
    this->height = height;
    this->kernelSettings = kernelSettings;
}

CellGrid::~CellGrid()
{
    if (device_gridMemory != nullptr)
    {
        cudaFree(device_gridMemory);
    }
}

void CellGrid::initialize_grid()
{
    // Allocate pitched memory for cells.
    cudaMallocPitch((void **)&device_gridMemory, &memoryPitch, width * sizeof(Cell), height);
    // Generate random cells on device.
    generate_random_population<<<kernelSettings.gridDimension, kernelSettings.blockDimension>>>(device_gridMemory, memoryPitch, width, height);
}
