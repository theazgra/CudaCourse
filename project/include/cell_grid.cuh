#pragma once
#include <cell.cuh>
#include <cuda_utils.cuh>
#include <image.h>
#include <time.h>

class CellGrid
{
private:
  // Grid dimensions.
  size_t width = 0;
  size_t height = 0;
  size_t textureWidth = 0;
  size_t textureHeight = 0;
  KernelSettings kernelSettings;
  // Fitness.
  float lastPopulationFitness = 0.0f;

  // Memory
  size_t currPopPitch;
  size_t nextPopPitch;
  Cell *device_currPopMemory = nullptr;
  Cell *device_nextPopMemory = nullptr;

  void print_cell_grid(const Cell *data, const size_t pitch, bool fitness = false) const;

  void create_fitness_texture(const Image &fitnessImage);

public:
  CellGrid(const size_t width, const size_t height, KernelSettings kernelSettings);
  ~CellGrid();
  void initialize_grid(const Image &fitnessImage);
  void evolve(float &evolutionTime);
  float get_average_fitness(float &reduceTime) const;
};