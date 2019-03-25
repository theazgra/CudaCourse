#include "../../my_cuda.cu"

struct __align__(16) Time
{
    int hour;
    int minute;
    int second;

    void print() { printf("Time %i:%i:%i\n", hour, minute, second); }
};

constexpr int cedw = sizeof(Time);

__constant__ __device__ int dConstScalar;
__constant__ __device__ Time dTime;
__constant__ __device__ int dArray[5];

/*
    Try to write a simple code that will allocate and set a scalar value in the GPU constant memory.
    Copy the data back to HOST and check the value.
    Do the same with custom structure and then with some array.
*/

int main(int, char **)
{
    const int hConstScalar = 50;
    HANDLE_ERROR(cudaMemcpyToSymbol(static_cast<const void *>(&dConstScalar), &hConstScalar, sizeof(int), 0, cudaMemcpyHostToDevice));

    int scalarBack;
    HANDLE_ERROR(cudaMemcpyFromSymbol(&scalarBack, static_cast<const void *>(&dConstScalar), sizeof(int), 0, cudaMemcpyDeviceToHost));
    printf("Original %i, got back %i\n", hConstScalar, scalarBack);

    Time time = {};
    time.hour = 10;
    time.minute = 48;
    time.second = 22;

    HANDLE_ERROR(cudaMemcpyToSymbol(static_cast<const void *>(&dTime), &time, sizeof(Time)));
    Time timeBack = {};
    HANDLE_ERROR(cudaMemcpyFromSymbol(&timeBack, static_cast<const void *>(&dTime), sizeof(Time)));
    time.print();
    timeBack.print();

    int hostArr[5] = {5, 4, 3, 2, 1};
    cudaMemcpyToSymbol(dArray, hostArr, sizeof(int) * 5);
    int hostArrBack[5];
    cudaMemcpyFromSymbol(hostArrBack, dArray, sizeof(int) * 5);

    for (size_t i = 0; i < 5; i++)
    {
        assert(hostArr[i] == hostArrBack[i]);
        printf("%i vs %i\n", hostArr[i], hostArrBack[i]);
    }
}
