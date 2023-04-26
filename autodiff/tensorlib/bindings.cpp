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
        .def("onCPU", &Tensor::onCPU)
        .def("gpuFree", &Tensor::gpuFree);

    m.def("cpu_add", [](Tensor *a, Tensor *b) { return cpu_add(a, b); },
          py::return_value_policy::take_ownership);
    m.def("gpu_add", [](Tensor *a, Tensor *b) { return gpu_add(a, b); },
          py::return_value_policy::take_ownership);

    m.def("cpu_mul", [](Tensor *a, Tensor *b) { return cpu_mul(a, b); },
          py::return_value_policy::take_ownership);
    m.def("gpu_mul", [](Tensor *a, Tensor *b) { return gpu_mul(a, b); },
          py::return_value_policy::take_ownership);

    m.def("cpu_sum", [](Tensor *a, int axis) { return cpu_sum(a, axis); },
          py::return_value_policy::take_ownership);
    m.def("gpu_sum", [](Tensor *a, int axis) { return gpu_sum(a, axis); },
          py::return_value_policy::take_ownership);

    m.def("cpu_bct",
          [](Tensor *a, int axis, int dim) { return cpu_bct(a, axis, dim); },
          py::return_value_policy::take_ownership);
    m.def("gpu_bct",
          [](Tensor *a, int axis, int dim) { return gpu_bct(a, axis, dim); },
          py::return_value_policy::take_ownership);

    m.def("cpu_set_zero", [](Tensor *a) { return cpu_set_zero(a); },
          py::return_value_policy::take_ownership);
    m.def("gpu_set_zero", [](Tensor *a) { return gpu_set_zero(a); },
          py::return_value_policy::take_ownership);

    m.def("cpu_cpy", [](Tensor *a) { return cpu_cpy(a); },
          py::return_value_policy::take_ownership);
    m.def("gpu_cpy", [](Tensor *a) { return gpu_cpy(a); },
          py::return_value_policy::take_ownership);

    m.def("cpu_exp", [](Tensor *a) { return cpu_exp(a); },
          py::return_value_policy::take_ownership);
    m.def("gpu_exp", [](Tensor *a) { return gpu_exp(a); },
          py::return_value_policy::take_ownership);

    m.def("cpu_tsp", [](Tensor *a) { return cpu_tsp(a); },
          py::return_value_policy::take_ownership);
    m.def("gpu_tsp", [](Tensor *a) { return gpu_tsp(a); },
          py::return_value_policy::take_ownership);
}