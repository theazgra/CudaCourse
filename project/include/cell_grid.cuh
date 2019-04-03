#pragma once
#include <cell.cuh>
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
  size_t currPopMemoryPitch;
  size_t nextPopMemoryPitch;
  Cell *device_currPopMemory = nullptr;
  Cell *device_nextPopMemory = nullptr;

  void print_cell_grid() const;

public:
  CellGrid(const size_t width, const size_t height, KernelSettings kernelSettings);
  ~CellGrid();
  void initialize_grid();
  void evolve();
  float get_average_fitness() const;
};