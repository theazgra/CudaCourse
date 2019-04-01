#pragma once

#include <cuda_utils.cuh>
#include <cell.cuh>

static __global__ void generate_random_population(Cell *currentPopulation, const size_t memoryPitch, const size_t width, const size_t height)
{
    //TODO: Generate random population to currentPopulation
}

// This kernel will evole current population into new one.
static __global__ void evolve()
{
}
