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

Tensor *gpu_add(Tensor *a, Tensor *b) {
    a->onGpuAssert();
    b->onGpuAssert();
    py::buffer_info a_info = a->request();
    py::buffer_info b_info = b->request();

    if (a_info.shape.size() != b_info.shape.size()) {
        throw std::runtime_error("Dimentions don't match");
    }

    if (a_info.shape.size() != 2) {
        throw std::runtime_error("Only 2D tensors supported");
    }

    int dim0 = a_info.shape[0];
    int dim1 = a_info.shape[1];

    Tensor *result = createGPUTensor(dim0, dim1);

    const int threadsPerBlock = 512;
    int blocks = (result->size() + threadsPerBlock - 1) / threadsPerBlock;

    _add<<<blocks, threadsPerBlock>>>(a->dataGpu(), b->dataGpu(),
                                      result->dataGpu(), dim0, dim1);

    return result;
}