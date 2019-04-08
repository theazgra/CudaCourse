#include <device_launch_parameters.h>
#include <image.h>
#include <assert.h>
#include <cell_grid.cuh>

constexpr int NumberOfEvolutions = 1000;
constexpr int CellGridDimension = 10000;

int main(int argc, char const *argv[])
{
    KernelSettings ks = {};
    ks.blockDimension = dim3(ThreadsPerBlock, ThreadsPerBlock, 1);
    ks.gridDimension = dim3(get_number_of_parts(CellGridDimension, ThreadsPerBlock), get_number_of_parts(CellGridDimension, ThreadsPerBlock), 1);

    CellGrid grid(CellGridDimension, CellGridDimension, ks);
    Image fitnessImage = Image("../images/radial.png", ImageType_GrayScale_8bpp);
    grid.initialize_grid(fitnessImage);
    
    /*
    float avgFit = grid.get_average_fitness();
    printf("Before evolve: %.3f\n", avgFit);
    grid.evolve();
    avgFit = grid.get_average_fitness();
    printf("After evolve: %.3f\n", avgFit);
    */
    float fitness = 0.0;
    float lastFitness = -1.0f;
    uint iter = 0;
    double diff = 0;

    while (iter < NumberOfEvolutions && fitness != lastFitness)
    {
        lastFitness = fitness;
        ++iter;

        grid.evolve();
        fitness = grid.get_average_fitness();
        diff = fitness - lastFitness;
        printf("Finished iteration %u, fitness: %.6f\t %s%.6f\n", iter+1, fitness,
                diff>=0?"+":"-",diff);
    }

    return 0;
}
