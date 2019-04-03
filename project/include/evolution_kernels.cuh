#pragma once

#include <cuda_utils.cuh>
#include <device_launch_parameters.h>
#include <cell.cuh>

// This kernel will evole current population into new one.
__global__ static void get_finess_kernel(CellGridInfo *gridInfo, float *fitness)
{
}