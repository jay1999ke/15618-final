#include "blas.h"

py::array_t<float> cpu_add(py::array_t<float> a, py::array_t<float> b) {
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

    float *res_ptr = new float[dim1 * dim2];

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            res_ptr[i * dim1 + j] = a_ptr[i * dim1 + j] + b_ptr[i * dim1 + j];
        }
    }

    return py::array_t<float>(
        {dim1, dim2}, // shape
        {sizeof(float) * dim1,
         sizeof(float)}, // C-style contiguous strides for double
        res_ptr          // the data pointer
        );               
}