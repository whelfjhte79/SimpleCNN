#ifndef KERNEL_TEST_H
#define KERNEL_TEST_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>

/*
__global__ void sum_kernel(int a, int b, int* c) {
    for (int i = 0; i < 100; i++) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        printf("Ãâ·Â: %d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
        c[tid] = a + b;
    }
}
int sum_cuda(int a, int b, int* c) {
    int* f;
    cudaMalloc((void**)&f, sizeof(int) * 1);
    cudaMemcpy(f, c, sizeof(int) * 1, cudaMemcpyHostToDevice);

    sum_kernel<<<128, 64>>> (a, b, f);
    cudaMemcpy(c, f, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    cudaFree(f);
    return true;
}
*/

#endif //KERNEL_TEST_H