#include "tensor.h"

Tensor *createGPUTensor(size_t rows, size_t cols) {
    Tensor *tensor = new Tensor(rows, cols);
    tensor->_gpu();
    return tensor;
}

void Tensor::setOnGpu(bool val) { on_gpu = val; }

void Tensor::_gpu() {
    if (gpu_data == nullptr)
        cudaMalloc(&gpu_data, size());
    setOnGpu(true);
}

void Tensor::gpu() {
    _gpu();
    cudaMemcpy(dataGpu(), data(), size(), cudaMemcpyHostToDevice);
}

void Tensor::cpu() {
    if (dataGpu() != nullptr)
        cudaMemcpy(data(), dataGpu(), size(), cudaMemcpyDeviceToHost);
}

void Tensor::gpuFree() {
    if (dataGpu() != nullptr)
        cudaFree(dataGpu());
    gpu_data = nullptr;
    setOnGpu(false);
}

void gpu_set_zero(Tensor *a) {
    cudaMemset(a->dataGpu(), 0, a->size());
}

Tensor *gpu_add(Tensor *a, Tensor *b) {
    a->onGpuAssert();
    b->onGpuAssert();
    a->sameShapeAssert(b);

    py::buffer_info a_info = a->request();
    py::buffer_info b_info = b->request();

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    Tensor *result = createGPUTensor(dim0, dim1);

    const int threadsPerBlock = 512;
    int blocks = (result->size() + threadsPerBlock - 1) / threadsPerBlock;

    _add<<<blocks, threadsPerBlock>>>(a->dataGpu(), b->dataGpu(),
                                      result->dataGpu(), dim0, dim1);

    return result;
}

Tensor *gpu_mul(Tensor *a, Tensor *b) {
    a->onGpuAssert();
    b->onGpuAssert();
    a->sameShapeAssert(b);
    
    py::buffer_info a_info = a->request();
    py::buffer_info b_info = b->request();

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    Tensor *result = createGPUTensor(dim0, dim1);

    const int threadsPerBlock = 512;
    int blocks = (result->size() + threadsPerBlock - 1) / threadsPerBlock;

    _mul<<<blocks, threadsPerBlock>>>(a->dataGpu(), b->dataGpu(),
                                      result->dataGpu(), dim0, dim1);

    return result;
}

Tensor *gpu_sum(Tensor *a, int axis) {
    a->onGpuAssert();

    py::buffer_info a_info = a->request();

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

    Tensor *result = createGPUTensor(res_dim0, res_dim1);

    const int threadsPerBlock = 512;
    int blocks = (result->size() + threadsPerBlock - 1) / threadsPerBlock;

    _sum<<<blocks, threadsPerBlock>>>(a->dataGpu(), result->dataGpu(), dim0,
                                      dim1, axis);

    return result;
}

Tensor *gpu_bct(Tensor *a, int axis, int dim) {
    a->onGpuAssert();

    py::buffer_info a_info = a->request();

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

    Tensor *result = createGPUTensor(res_dim0, res_dim1);

    const int threadsPerBlock = 512;
    int blocks = (result->size() + threadsPerBlock - 1) / threadsPerBlock;

    _bct<<<blocks, threadsPerBlock>>>(a->dataGpu(), result->dataGpu(), res_dim0,
                                      res_dim1, axis);

    return result;
}