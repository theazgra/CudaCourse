#include <device_launch_parameters.h>
#include <image.h>
#include <assert.h>
#include <cell_grid.cuh>

constexpr int NumberOfEvolutions = 5000;
constexpr int CellGridDimension = 1000; //10000
constexpr size_t CellCount = CellGridDimension * CellGridDimension;

int main(int argc, char const *argv[])
{
    printf("Cell count: %lu\n", CellCount);
    KernelSettings ks = {};
    ks.blockDimension = dim3(ThreadsPerBlock, ThreadsPerBlock, 1);
    ks.gridDimension = dim3(get_number_of_parts(CellGridDimension, ThreadsPerBlock), get_number_of_parts(CellGridDimension, ThreadsPerBlock), 1);

    CellGrid grid(CellGridDimension, CellGridDimension, ks);
    Image fitnessImage = Image("/home/mor0146/github/CudaCourse/project/images/radial16bit_2.png", ImageType_GrayScale_16bpp);
    grid.initialize_grid(fitnessImage);

    float fitness = 0.0;
    float lastFitness = -1.0f;
    uint iter = 0;
    double diff = 0;
    double averageEvolveTime = 0.0f;
    double averageFitnessTime = 0.0f;
    size_t sameFitnessValue = 0;
    while (iter < NumberOfEvolutions && sameFitnessValue < 5)
    {
        float evolveTime, fitnessTime;

        lastFitness = fitness;
        ++iter;

        grid.evolve(evolveTime);
        averageEvolveTime += evolveTime;

        fitness = grid.get_average_fitness(fitnessTime);
        averageFitnessTime += fitnessTime;

        diff = fitness - lastFitness;

        if (fitness == lastFitness)
            sameFitnessValue++;
        else
            sameFitnessValue = 0;

        printf("Finished iteration %u, fitness: %.6f\t %.6f\n", iter + 1, fitness, diff); //diff >= 0 ? "+" : "-",
    }

    averageEvolveTime /= (double)iter;
    averageFitnessTime /= (double)iter;

    printf("Average evolve time: %f ms\n", averageEvolveTime);
    printf("Average fitness time: %f ms\n", averageFitnessTime);

    return 0;
}
