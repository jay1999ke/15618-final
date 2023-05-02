#include "tensor.h"

std::string Tensor::repr() {
    if (on_gpu)
        maintain();
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

void cpu_set_zero(Tensor *a) { memset(a->data(), 0, a->size()); }

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

Tensor *cpu_sub(Tensor *a, Tensor *b) {
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
            res_ptr[i * dim1 + j] = a_ptr[i * dim1 + j] - b_ptr[i * dim1 + j];
        }
    }

    return result;
}

Tensor *cpu_neg(Tensor *a) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    auto a_ptr = static_cast<float *>(a_info.ptr);

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    float *res_ptr = result->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] = -a_ptr[i * dim1 + j];
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

Tensor *cpu_div(Tensor *a, Tensor *b) {
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
            res_ptr[i * dim1 + j] = a_ptr[i * dim1 + j] / b_ptr[i * dim1 + j];
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

Tensor *cpu_cpy(Tensor *a) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];
    size_t size = a->size();

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap

    memcpy(result->data(), a->data(), size);

    return result;
}

Tensor *cpu_exp(Tensor *a) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];
    size_t size = a->size();

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    auto res_ptr = result->data();
    auto a_ptr = a->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] = std::exp(a_ptr[i * dim1 + j]);
        }
    }

    return result;
}

Tensor *cpu_log(Tensor *a) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];
    size_t size = a->size();

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    auto res_ptr = result->data();
    auto a_ptr = a->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] = std::log(a_ptr[i * dim1 + j]);
        }
    }

    return result;
}

Tensor *cpu_tsp(Tensor *a) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];
    size_t size = a->size();

    Tensor *result = new Tensor(dim1, dim0); // create an object on the heap
    auto res_ptr = result->data();
    auto a_ptr = a->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[j * dim0 + i] = a_ptr[i * dim1 + j];
        }
    }

    return result;
}

Tensor *cpu_pow(Tensor *a, float val) {
    a->onCpuAssert();

    py::buffer_info a_info = a->request();

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];
    size_t size = a->size();

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    auto res_ptr = result->data();
    auto a_ptr = a->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] = std::pow(a_ptr[i * dim1 + j], val);
        }
    }

    return result;
}

Tensor *cpu_relu(Tensor *a) {
    a->onCpuAssert();

    int dim0 = a->rows();
    int dim1 = a->cols();
    size_t size = a->size();

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    auto res_ptr = result->data();
    auto a_ptr = a->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] =
                a_ptr[i * dim1 + j] > 0 ? a_ptr[i * dim1 + j] : 0;
        }
    }

    return result;
}

Tensor *cpu_relu_grad(Tensor *a, Tensor *grad) {
    a->onCpuAssert();
    grad->onCpuAssert();
    a->sameShapeAssert(grad);

    int dim0 = a->rows();
    int dim1 = a->cols();
    size_t size = a->size();

    Tensor *result = new Tensor(dim0, dim1); // create an object on the heap
    auto res_ptr = result->data();
    auto a_ptr = a->data();
    auto grad_ptr = grad->data();

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            res_ptr[i * dim1 + j] =
                a_ptr[i * dim1 + j] > 0 ? grad_ptr[i * dim1 + j] : 0;
        }
    }

    return result;
}

Tensor *cpu_matmul(Tensor *a, Tensor *b) {
    a->onCpuAssert();
    b->onCpuAssert();

    if (a->cols() != b->rows())
        throw std::runtime_error("Incompatible shape for matmul");

    int m = a->rows();
    int n = a->cols();
    int k = b->cols();

    Tensor *result = new Tensor(m, k); // create an object on the heap
    float *res_ptr = result->data();

    float *a_ptr = a->data();
    float *b_ptr = b->data();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            float dot_prod = 0;
            for (int l = 0; l < n; ++l) {
                dot_prod += a_ptr[i * n + l] * b_ptr[l * k + j];
            }
            res_ptr[i * k + j] = dot_prod;
        }
    }

    return result;
}