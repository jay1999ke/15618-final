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

    auto result = py::array(py::buffer_info(
        nullptr,       /* Pointer to data (nullptr -> ask NumPy to allocate!) */
        sizeof(float), /* Size of one item */
        py::format_descriptor<float>::value, /* Buffer format */
        a_info.ndim,                         /* How many dimensions? */
        {dim1, dim2},   /* Number of elements for each dimension */
        {sizeof(float), sizeof(float)} /* Strides for each dimension */
        ));

    py::buffer_info result_info = result.request();
    auto res_ptr = static_cast<float *>(result_info.ptr);

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            res_ptr[i * dim1 + j] = a_ptr[i * dim1 + j] + b_ptr[i * dim1 + j];
        }
    }
    return result;
}