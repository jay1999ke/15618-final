#include "tensor.h"

PYBIND11_MODULE(tensorlib, m) {
    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer([](Tensor &m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(float),
                                   py::format_descriptor<float>::format(), 2,
                                   {m.rows(), m.cols()},
                                   {sizeof(float) * m.cols(), sizeof(float)});
        })
        .def(py::init<size_t, size_t>())
        .def(py::init<py::array_t<float>>())
        .def("data", &Tensor::data)
        .def("rows", &Tensor::rows)
        .def("cols", &Tensor::cols)
        .def("__repr__", &Tensor::repr)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("gpuFree", &Tensor::gpuFree);

    m.def("cpu_add", [](Tensor a, Tensor b) { return cpu_add(a, b); },
          py::return_value_policy::take_ownership);
    m.def("gpu_add", [](Tensor a, Tensor b) { return gpu_add(a, b); },
          py::return_value_policy::take_ownership);
}