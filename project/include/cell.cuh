#pragma once
#include <cuda_runtime_api.h>
#include <cassert>

struct Cell;
struct CellGridInfo
{
    Cell *data;
    size_t pitch;
    size_t width;
    size_t height;
};

struct __align__(16) Cell
{
    int x = -1;
    int y = -1;
    float fitness = 0;

    __device__ __host__ Cell()
    {
    }

    __device__ Cell(int _x, int _y) : x(_x), y(_y)
    {
    }

    // Produce offspring.
    __device__ Cell(const Cell *parentA, const Cell *parentB)
    {
        x = (parentA->x + parentB->x) / 2;
        y = (parentA->y + parentB->y) / 2;
    }

    /*
    __device__ Cell *find_partner(const CellGridInfo *grid, const uint tIdX, const uint tIdY)
    {
        Cell *bestPartner = nullptr;

        // L5 cross
        Cell *topCell = ((Cell *)((char *)grid->data + mod(tIdY - 1, grid->height) * grid->pitch) + tIdX);
        Cell *bottomCell = ((Cell *)((char *)grid->data + mod(tIdY + 1, grid->height) * grid->pitch) + tIdX);
        Cell *leftCell = ((Cell *)((char *)grid->data + tIdY * grid->pitch) + mod(tIdX - 1, grid->width));
        Cell *rightCell = ((Cell *)((char *)grid->data + tIdY * grid->pitch) + mod(tIdX + 1, grid->width));

        bestPartner = topCell;
        if (bottomCell->fitness > bestPartner->fitness)
            bestPartner = bottomCell;
        if (leftCell->fitness > bestPartner->fitness)
            bestPartner = leftCell;
        if (rightCell->fitness > bestPartner->fitness)
            bestPartner = rightCell;

        assert(bestPartner != nullptr);

        return bestPartner;
    }
    */

    __device__ void random_mutation()
    {
        //TODO: Do we want to apply random mutation?
    }
};