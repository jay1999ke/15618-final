#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

namespace py = pybind11;

py::array_t<float> cpu_add(py::array_t<float> a, py::array_t<float> b);
py::array_t<float> gpu_add(py::array_t<float> a, py::array_t<float> b);