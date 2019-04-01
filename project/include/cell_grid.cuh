#pragma once
#include <evolution_kernels.cuh>

class CellGrid
{
  private:
    // Grid dimensions.
    size_t width = 0;
    size_t height = 0;
    KernelSettings kernelSettings;
    // Fitness.
    float lastPopulationFitness = 0.0f;
    // Memory
    size_t memoryPitch;
    Cell *device_gridMemory = nullptr;

  public:
    CellGrid(const size_t width, const size_t height, KernelSettings kernelSettings);
    ~CellGrid();
    void initialize_grid();
};