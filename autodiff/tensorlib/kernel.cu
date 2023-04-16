#include "tensor.h"

__global__ void _add(float *a, float *b, float *res, int dim0, int dim1) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim0 * dim1) {
        res[idx] = a[idx] + b[idx];
    }
}

__global__ void _mul(float *a, float *b, float *res, int dim0, int dim1) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim0 * dim1) {
        res[idx] = a[idx] * b[idx];
    }
}

__global__ void _sum(float *a, float *res, int dim0, int dim1, int axis) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int res_dim0, res_dim1;
    if (axis == 0) {
        res_dim0 = 1;
        res_dim1 = dim1;
    } else {
        res_dim0 = dim0;
        res_dim1 = 1;
    }

    if (idx < res_dim0 * res_dim1) {
        float temp_result = 0;
        if (axis == 0) {
            for (int i = 0; i < dim0; i++) {
                temp_result += a[i * dim1 + idx];
            }
        } else {
            for (int j = 0; j < dim1; j++) {
                temp_result += a[idx * dim1 + j];
            }
        }
        res[idx] = temp_result;
    }
}