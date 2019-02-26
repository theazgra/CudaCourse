#include "../../my_cuda.cu"
#include "Array2D.h"

constexpr int N = 300000;
constexpr int M = 15;
constexpr uint THREADS_PER_BLOCK = 256;

constexpr uint blockDim = 8;
constexpr uint rowCount = 10;
constexpr uint colCount = 5;

int main(void)
{
    /*
    Create a column matrix m[mRows,mCols] containing the numbers 0 1 2 3 ...
    The data should be well alligned in the page-locked memory.
    The matrix should be filled in CUDA kernel.
    You must use a Pitch CUDA memory with appropriate alignment. Moreover you must use 2D grid of 2D blocks of size 8x8.
    Increment the values of the matrix.
    Finally, copy the matrix to HOST using cudaMemcpy2D function.
    */

    int *dMatrix;
    size_t pitch = 0;

    cudaMallocPitch((void **)&dMatrix, &pitch, rowCount * sizeof(int), colCount);
    printf("Pitch: %lu \n", pitch);

    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize(get_number_of_parts(rowCount, blockDim), get_number_of_parts(colCount, blockDim));
    //dim3 gridSize(get_number_of_parts(colCount, 8), get_number_of_parts(rowCount, 8));

    return 0;
}
