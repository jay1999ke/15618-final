#include "blas.h"

__global__ void _add(float *a, float *b, float *res,int dim1, int dim2) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim1 * dim2) {
        res[idx] = a[idx] + b[idx];
    }
}

py::array_t<float> gpu_add(py::array_t<float> a, py::array_t<float> b) {
    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    if (a_info.shape.size() != b_info.shape.size()) {
        throw std::runtime_error("Dimentions don't match");
    }

    if (a_info.shape.size() != 2) {
        throw std::runtime_error("Only 2D tensors supported");
    }

    auto a_ptr = static_cast<float *>(a_info.ptr);
    auto b_ptr = static_cast<float *>(b_info.ptr);

    int dim1 = a_info.shape[0];
    int dim2 = a_info.shape[1];

    auto result = py::array(py::buffer_info(
        nullptr,       /* Pointer to data (nullptr -> ask NumPy to allocate!) */
        sizeof(float), /* Size of one item */
        py::format_descriptor<float>::value, /* Buffer format */
        a_info.ndim,                         /* How many dimensions? */
        {dim1, dim2}, /* Number of elements for each dimension */
        {sizeof(float) * dim1, sizeof(float)} /* Strides for each dimension */
        ));

    py::buffer_info result_info = result.request();
    auto res_ptr = static_cast<float *>(result_info.ptr);

    int size = dim1 * dim2;

    const int threadsPerBlock = 512;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    float *cuda_a;
    float *cuda_b;
    float *cuda_res;

    cudaMalloc(&cuda_a, size * sizeof(float)); 
    cudaMalloc(&cuda_b, size * sizeof(float)); 
    cudaMalloc(&cuda_res, size * sizeof(float)); 

    cudaMemcpy(cuda_a, a_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b_ptr, size * sizeof(float), cudaMemcpyHostToDevice);

    _add<<<blocks, threadsPerBlock>>>(cuda_a, cuda_b, cuda_res, dim1, dim2);

    cudaMemcpy(res_ptr, cuda_res, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_res);

    return result;
}