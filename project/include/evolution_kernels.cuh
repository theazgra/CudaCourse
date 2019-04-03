#pragma once

#include <cuda_utils.cuh>
#include <device_launch_parameters.h>
#include <cell.cuh>

// This kernel will evole current population into new one.
__global__ static void evolve_kernel(CellGridInfo *currentPopulation, CellGridInfo *nextPopulation)
{
    //TODO: Each cell will find its neighborhood, reorder it to find two most fit cells.
    //  Texture would be nice so we wouldn't have to check borders
    int tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;

    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    while (tIdX < currentPopulation->width)
    {
        tIdY = (blockIdx.y * blockDim.y) + threadIdx.y;
        while (tIdY < currentPopulation->height)
        {
            //TODO: Get cell from pitched memory.
            Cell *c = nullptr;
            //TODO: Do we allow different neighborhoods?
            Cell neighborhood[8];
            c->get_sorted_neighborhood(currentPopulation, tIdX, tIdY, neighborhood);
            Cell offspring(&neighborhood[0], &neighborhood[1]);
            offspring.random_mutation();

            //TODO: Which cell do we replace in next population? Worst in neighborhood, one parent?
            //      Replace the cell in next population.

            tIdY += strideY;
        }
        tIdX += strideX;
    }
}

// This kernel will evole current population into new one.
__global__ static void get_finess_kernel(CellGridInfo *gridInfo, float *fitness)
{
}