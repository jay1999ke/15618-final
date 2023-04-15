#include "blas.h"

py::array_t<float> cpu_add(py::array_t<float> a, py::array_t<float> b);

PYBIND11_MODULE(blas, m) {
    m.def("cpu_add", [](py::array_t<float> a,
                        py::array_t<float> b) { return cpu_add(a, b); },
          py::return_value_policy::take_ownership);
}