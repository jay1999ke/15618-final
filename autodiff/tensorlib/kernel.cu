#include "tensor.h"

__global__ void _add(float *a, float *b, float *res, int dim0, int dim1) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim0 * dim1) {
        res[idx] = a[idx] + b[idx];
    }
}