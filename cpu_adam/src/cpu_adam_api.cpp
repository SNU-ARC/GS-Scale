#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quat_to_rotmat", &quat_to_rotmat, py::call_guard<py::gil_scoped_release>());
    m.def("persp_proj", &persp_proj, py::call_guard<py::gil_scoped_release>());
    m.def("world_to_cam", &world_to_cam, py::call_guard<py::gil_scoped_release>());
    m.def("calculate_update_ids", &calculate_update_ids, py::call_guard<py::gil_scoped_release>());
    m.def("update_counter", &update_counter, py::call_guard<py::gil_scoped_release>());
    m.def("adam_deferred_update", &adam_deferred_update, py::call_guard<py::gil_scoped_release>());
    m.def("adam_for_next_with_counter", &adam_for_next_with_counter, py::call_guard<py::gil_scoped_release>());
    m.def("sparse_adam", &sparse_adam, py::call_guard<py::gil_scoped_release>());
    m.def("adam_for_next", &adam_for_next, py::call_guard<py::gil_scoped_release>());
    m.def("index_copy", &index_copy, py::call_guard<py::gil_scoped_release>());
}

