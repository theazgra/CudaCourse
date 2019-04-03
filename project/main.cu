#include <device_launch_parameters.h>
#include <image.h>
#include <assert.h>
#include <cell_grid.cuh>

constexpr int NumberOfEvolutions = 500;
constexpr int BlockDimensionSize = 16;
constexpr int CellGridDimension = 15;

int main(int argc, char const *argv[])
{
    KernelSettings ks = {};
    ks.blockDimension = dim3(BlockDimensionSize, BlockDimensionSize, 1);
    ks.gridDimension = dim3(get_number_of_parts(CellGridDimension, BlockDimensionSize), get_number_of_parts(CellGridDimension, BlockDimensionSize), 1);

    CellGrid grid(CellGridDimension, CellGridDimension, ks);
    Image fitnessImage = Image("../test15x15.png", ImageType_GrayScale_8bpp);
    grid.initialize_grid(fitnessImage);

    grid.evolve();

    // for (size_t evolutionStep = 0; evolutionStep < NumberOfEvolutions; evolutionStep++)
    // {
    //     grid.evolve();
    //     float popFitness = grid.get_average_fitness();
    //     printf("Fitness of current population: %5f.5\n", popFitness);
    // }

    return 0;
}
