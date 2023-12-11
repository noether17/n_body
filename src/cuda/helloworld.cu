#include "stdio.h"

#define WARPSIZE 32

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x;
    if (tid < WARPSIZE)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void)
{
    int a[WARPSIZE];
    int b[WARPSIZE];
    int c[WARPSIZE];
    for (int i = 0; i < WARPSIZE; i++)
    {
        a[i] = i;
        b[i] = i + WARPSIZE;
    }

    int* dev_a;
    int* dev_b;
    int* dev_c;
    cudaMalloc((void**)&dev_a, WARPSIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, WARPSIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, WARPSIZE * sizeof(int));
    cudaMemcpy(dev_a, a, WARPSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, WARPSIZE * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, WARPSIZE>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, WARPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < WARPSIZE; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}