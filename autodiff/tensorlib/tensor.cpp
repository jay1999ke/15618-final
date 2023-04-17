#include "tensor.h"

std::string Tensor::repr() {
    if (on_gpu)
        cpu();
    std::string str = "tensor([";
    for (int i = 0; i < dim0; i++) {
        if (i != 0)
            str += ",\n        ";
        str += "[";
        for (int j = 0; j < dim1; j++) {
            if (j != 0)
                str += ", ";
            std::ostringstream strs;
            strs << std::fixed << std::setprecision(3)
                 << cpu_data[i * dim1 + j];
            str += strs.str();
        }
        str += "]";
    }
    if (on_gpu)
        str += "], gpu = True)";
    else
        str += "])";
    return str;
};

Tensor *cpu_add(Tensor *a, Tensor *b) {
    a->onCpuAssert();
    b->onCpuAssert();
    a->sameShapeAssert(b);

    py::buffer_info a_info = a->request();
    py::buffer_info b_info = b->request();

    auto a_ptr = static_cast<float *>(a_info.ptr);
    auto b_ptr = static_cast<float *>(b_info.ptr);

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    float *res_ptr = result->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] = a_ptr[i * dim1 + j] + b_ptr[i * dim1 + j];
        }
    }

    return result;
}

Tensor *cpu_mul(Tensor *a, Tensor *b) {
    a->onCpuAssert();
    b->onCpuAssert();
    a->sameShapeAssert(b);

    py::buffer_info a_info = a->request();
    py::buffer_info b_info = b->request();

    auto a_ptr = static_cast<float *>(a_info.ptr);
    auto b_ptr = static_cast<float *>(b_info.ptr);

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    float *res_ptr = result->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] = a_ptr[i * dim1 + j] * b_ptr[i * dim1 + j];
        }
    }

    return result;
}

Tensor *cpu_sum(Tensor *a, int axis) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    auto a_ptr = static_cast<float *>(a_info.ptr);

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    int res_dim0, res_dim1;

    if (axis == 0) {
        res_dim0 = 1;
        res_dim1 = dim1;
    } else if (axis == 1) {
        res_dim0 = dim0;
        res_dim1 = 1;
    } else {
        throw std::runtime_error("Invalid sum axis");
    }

    Tensor *result = new Tensor(res_dim0, res_dim1, true);
    float *res_ptr = result->data();

    if (axis == 0) {
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                res_ptr[j] += a_ptr[i * dim1 + j];
            }
        }
    } else {
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                res_ptr[i] += a_ptr[i * dim1 + j];
            }
        }
    }

    return result;
}

Tensor *cpu_bct(Tensor *a, int axis, int dim) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    auto a_ptr = static_cast<float *>(a_info.ptr);

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    int res_dim0, res_dim1;

    if (axis == 0) {
        res_dim0 = dim;
        res_dim1 = dim1;
    } else if (axis == 1) {
        res_dim0 = dim0;
        res_dim1 = dim;
    } else {
        throw std::runtime_error("Invalid sum axis");
    }

    Tensor *result = new Tensor(res_dim0, res_dim1);
    float *res_ptr = result->data();

    if (axis == 0) {
        for (int i = 0; i < res_dim0; i++) {
            for (int j = 0; j < res_dim1; j++) {
                res_ptr[i * res_dim1 + j] = a_ptr[j];
            }
        }
    } else {
        for (int i = 0; i < res_dim0; i++) {
            for (int j = 0; j < res_dim1; j++) {
                res_ptr[i * res_dim1 + j] = a_ptr[i];
            }
        }
    }

    return result;
}