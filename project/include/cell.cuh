#pragma once
#include <cuda_runtime_api.h>

struct CellGridInfo;

struct Cell
{
    int x = -1;
    int y = -1;

    __device__ __host__ Cell()
    {
    }

    __device__ Cell(int _x, int _y) : x(_x), y(_y)
    {
    }

    // Produce offspring.
    __device__ Cell(const Cell *parentA, const Cell *parentB)
    {
    }

    __device__ void get_sorted_neighborhood(const CellGridInfo *grid, int x, int y, Cell *neighborhood)
    {
    }

    __device__ void random_mutation()
    {
    }
};

struct CellGridInfo
{
    Cell *data;
    size_t pitch;
    size_t width;
    size_t height;
};
