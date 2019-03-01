#include "../../my_cuda.cu"
#include "Array2D.h"

constexpr uint desiredBlockDim = 8;
constexpr uint rowCount = 20;
constexpr uint colCount = 10;

__global__ void kernel_init_column_matrix(int *matrix, size_t pitch)
{
    uint xOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint yOffset = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (xOffset < colCount && yOffset < rowCount)
    {
        int *memoryRow = (int *)((char *)matrix + (xOffset * pitch));
        memoryRow[yOffset] = (xOffset * rowCount) + yOffset;
    }
}

__global__ void kernel_incerement(int *matrix, size_t pitch)
{
    // uint skip = gridDim.x * THREADS_PER_BLOCK;
    // while (offset < N)
    // {
    //     c[offset] = a[offset] + b[offset];
    //     offset += skip;
    // }

    uint xOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint yOffset = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (xOffset < colCount && yOffset < rowCount)
    {
        int *memoryRow = (int *)((char *)matrix + (xOffset * pitch));
        memoryRow[yOffset]++;
    }
}

int main(void)
{
    /*
    [ ] Create a column matrix m[mRows,mCols] containing the numbers 0 1 2 3 ...
            The data should be well alligned in the page-locked memory.
            The matrix should be filled in CUDA kernel.
    [ ] You must use a Pitch CUDA memory with appropriate alignment. Moreover you must use 2D grid of 2D blocks of size 8x8.
    [ ] Increment the values of the matrix.
    [ ] Finally, copy the matrix to HOST using cudaMemcpy2D function.
    */

    int *dMatrix;
    size_t pitch = 0;

    HANDLE_ERROR(cudaMallocPitch((void **)&dMatrix, &pitch, rowCount * sizeof(int), colCount));

    uint gridXDim = get_number_of_parts(rowCount, desiredBlockDim);
    uint gridYDim = get_number_of_parts(colCount, desiredBlockDim);
    dim3 blockSize(desiredBlockDim, desiredBlockDim);
    dim3 gridSize(gridXDim, gridYDim);

    printf("Pitch:\t%lu \n", pitch);
    printf("Grid:\t%u x %u\n", gridXDim, gridYDim);
    printf("Block:\t%u x %u\n", desiredBlockDim, desiredBlockDim);
    printf("Matrix:\t%u x %u\n", colCount, rowCount);

    CUDA_TIMED_BLOCK_START("Column Matrix Initialization");
    kernel_init_column_matrix<<<gridSize, blockSize>>>(dMatrix, pitch);
    CUDA_TIMED_BLOCK_END;

    HANDLE_ERROR(cudaPeekAtLastError());

    // kernel_incerement<<<gridSize, blockSize>>>(dMatrix, pitch);
    // HANDLE_ERROR(cudaPeekAtLastError());

    int result[colCount][rowCount];
    HANDLE_ERROR(cudaMemcpy2D(result, rowCount * sizeof(int), dMatrix, pitch, rowCount * sizeof(int), colCount, cudaMemcpyDeviceToHost));
    for (uint row = 0; row < rowCount; row++)
    {
        for (uint col = 0; col < colCount; col++)
        {
            printf("%4i ", result[col][row]);
        }
        printf("\n");
    }
    cudaFree(dMatrix);

    //CUDA_TIMED_FUNCTION(printf("Hello World from macro\n."));
    return 0;
}
