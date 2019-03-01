#include "../../my_cuda.cu"
#include "Array2D.h"

constexpr uint desiredBlockDim = 8;
constexpr uint rowCount = 16000;
constexpr uint colCount = 16000;

__global__ void kernel_init_column_matrix(int *matrix, size_t pitch)
{
    uint xOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint yOffset = (blockIdx.y * blockDim.y) + threadIdx.y;
    // uint skipX = gridDim.x * blockDim.x;
    // uint skipY = gridDim.y * blockDim.y;

    if (xOffset < colCount && yOffset < rowCount)
    {
        int *memoryRow = (int *)((char *)matrix + (xOffset * pitch));
        memoryRow[yOffset] = (xOffset * rowCount) + yOffset;
    }
    // while (xOffset < colCount && yOffset < rowCount)
    // {

    //     xOffset += skipX;
    //     yOffset += skipY;
    // }
}

__global__ void kernel_incerement(int *matrix, size_t pitch)
{
    uint xOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint yOffset = (blockIdx.y * blockDim.y) + threadIdx.y;

    uint skipX = gridDim.x * blockDim.x;
    uint skipY = gridDim.y * blockDim.y;

    while (xOffset < colCount && yOffset < rowCount)
    {
        int *memoryRow = (int *)((char *)matrix + (xOffset * pitch));
        memoryRow[yOffset]++;

        xOffset += skipX;
        yOffset += skipY;
    }
}

int main(void)
{
    int *dMatrix;
    size_t pitch = 0;

    HANDLE_ERROR(cudaMallocPitch((void **)&dMatrix, &pitch, rowCount * sizeof(int), colCount));

    uint gridXDim = 1 + get_number_of_parts(rowCount, desiredBlockDim);
    uint gridYDim = 1 + get_number_of_parts(colCount, desiredBlockDim);
    dim3 blockSize(desiredBlockDim, desiredBlockDim);
    dim3 gridSize(gridXDim, gridYDim);

    printf("Pitch:\t%lu \n", pitch);
    printf("Grid:\t%u x %u\n", gridXDim, gridYDim);
    printf("Block:\t%u x %u\n", desiredBlockDim, desiredBlockDim);
    printf("Matrix:\t%u x %u\n", rowCount, colCount);

    CUDA_TIMED_BLOCK_START("Column Matrix Initialization And Incement.");
    kernel_init_column_matrix<<<gridSize, blockSize>>>(dMatrix, pitch);
    // kernel_incerement<<<gridSize, blockSize>>>(dMatrix, pitch);
    HANDLE_ERROR(cudaPeekAtLastError());
    CUDA_TIMED_BLOCK_END;

    size_t resultSize = colCount * rowCount * sizeof(int);
    int **result = static_cast<int **>(malloc(resultSize));
    HANDLE_ERROR(cudaMemcpy2D(result, rowCount * sizeof(int), dMatrix, pitch, rowCount * sizeof(int), colCount, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dMatrix));

#ifndef NDEBUG
    for (uint row = 0; row < rowCount; row++)
    {
        int *rowMem = (int *)(result + (int)row);
        for (uint col = 0; col < colCount; col++)
        {
            printf("%4i ", *(rowMem + col));
        }
        printf("\n");
    }
#endif

    free(result);

    return 0;
}
