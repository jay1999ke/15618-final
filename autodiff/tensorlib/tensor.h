#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

class Tensor {
  public:
    Tensor(size_t rows, size_t cols) : dim0(rows), dim1(cols) {
        cpu_data = new float[rows * cols];
        on_gpu = false;
        gpu_data = nullptr;
    }

    // creating an object from numpy array (expected to be on cpu)
    Tensor(py::array_t<float> numpy) {
        py::buffer_info info = numpy.request();
        auto orig_data = info.ptr;
        assert(info.shape.size() == 2);
        dim0 = info.shape[0];
        dim1 = info.shape[1];
        on_gpu = false;
        gpu_data = nullptr;

        cpu_data = new float[rows() * cols()];
        memcpy(cpu_data, orig_data, size());
    }

    ~Tensor() {
        delete[] cpu_data;
        if (on_gpu)
            gpuFree();
    };

    // buffer ops
    float *data() { return cpu_data; }
    size_t rows() const { return dim0; }
    size_t cols() const { return dim1; }

    // python api
    void gpu();
    void cpu();
    void gpuFree();
    bool onCPU() { return !on_gpu; };
    std::string repr();
    py::buffer_info request() {
        return py::buffer_info(
            data(), sizeof(float), py::format_descriptor<float>::format(), 2,
            {rows(), cols()}, {sizeof(float) * cols(), sizeof(float)});
    };

    // cpp internal ops
    float *dataGpu() { return gpu_data; }
    void _gpu();
    void setOnGpu(bool);
    size_t size() const { return dim0 * dim1 * sizeof(float); };
    void onCpuAssert() { assert(on_gpu == false); };
    void onGpuAssert() { assert(on_gpu == true); };

  private:
    size_t dim0, dim1;
    float *cpu_data;
    bool on_gpu;
    float *gpu_data;
};

// gpu helpers
Tensor *createGPUTensor(size_t rows, size_t cols);

// cpu arith ops
Tensor *cpu_add(Tensor *a, Tensor *b);
Tensor *cpu_mul(Tensor *a, Tensor *b);

// gpu arith ops
Tensor *gpu_add(Tensor *a, Tensor *b);
Tensor *gpu_mul(Tensor *a, Tensor *b);

// kernels
__global__ void _add(float *a, float *b, float *res, int dim0, int dim1);
__global__ void _mul(float *a, float *b, float *res, int dim0, int dim1);
