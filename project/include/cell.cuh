#pragma once
#include "neighborhood_type.h"
#include <cassert>
#include <cuda_utils.cuh>

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

    template <NeighborhoodType neigh>
    __device__ void get_neighborhood(const uint tIdX, const uint tIdY,
                                     const CellGridInfo grid, Cell *neighborhood)
    {
        switch (neigh)
        {
        case NeighborhoodType_L5:
        {
            neighborhood[0] = grid.data[(mod(tIdY - 1, grid.height) * grid.width) + tIdX]; // Left
            neighborhood[1] = grid.data[(mod(tIdY + 1, grid.height) * grid.width) + tIdX]; // Right
            neighborhood[2] = grid.data[(tIdY * grid.width) + mod(tIdX - 1, grid.width)];  // Top
            neighborhood[3] = grid.data[(tIdY * grid.width) + mod(tIdX + 1, grid.width)];  // Bottom
        }
        break;
        case NeighborhoodType_L9:
        {
            neighborhood[0] = grid.data[(mod(tIdY - 1, grid.height) * grid.width) + tIdX]; // Left
            neighborhood[1] = grid.data[(mod(tIdY + 1, grid.height) * grid.width) + tIdX]; // Right
            neighborhood[2] = grid.data[(tIdY * grid.width) + mod(tIdX - 1, grid.width)];  // Top
            neighborhood[3] = grid.data[(tIdY * grid.width) + mod(tIdX + 1, grid.width)];  // Bottom

            neighborhood[4] = grid.data[(mod(tIdY - 2, grid.height) * grid.width) + tIdX]; // Left 2
            neighborhood[5] = grid.data[(mod(tIdY + 2, grid.height) * grid.width) + tIdX]; // Right 2
            neighborhood[6] = grid.data[(tIdY * grid.width) + mod(tIdX - 2, grid.width)];  // Top 2
            neighborhood[7] = grid.data[(tIdY * grid.width) + mod(tIdX + 2, grid.width)];  // Bottom 2
        }
        break;
        case NeighborhoodType_C9:
        {
            int fromRow = tIdY - 1;
            int toRow = tIdY + 2;
            int fromCol = tIdX - 1;
            int toCol = tIdX + 2;

            int i = 0;

            for (int row = fromRow; row < toRow; row++)
            {
                for (int col = fromCol; col < toCol; col++)
                {
                    neighborhood[i++] = grid.data[((mod(row, grid.height) * grid.width) + mod(col, grid.width))];
                }
            }
        }
        break;
        case NeighborhoodType_C13:
        {
            int fromRow = tIdY - 1;
            int toRow = tIdY + 2;
            int fromCol = tIdX - 1;
            int toCol = tIdX + 2;

            int i = 0;

            for (int row = fromRow; row < toRow; row++)
            {
                for (int col = fromCol; col < toCol; col++)
                {
                    neighborhood[i++] = grid.data[((mod(row, grid.height) * grid.width) + mod(col, grid.width))];
                }
            }

            neighborhood[i++] = grid.data[(mod(tIdY - 2, grid.height) * grid.width) + tIdX]; // Left 2
            neighborhood[i++] = grid.data[(mod(tIdY + 2, grid.height) * grid.width) + tIdX]; // Right 2
            neighborhood[i++] = grid.data[(tIdY * grid.width) + mod(tIdX - 2, grid.width)];  // Top 2
            neighborhood[i++] = grid.data[(tIdY * grid.width) + mod(tIdX + 2, grid.width)];  // Bottom 2
        }
        break;
        }
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

    __device__ void
    random_mutation()
    {
        //TODO: Do we want to apply random mutation?
    }
};