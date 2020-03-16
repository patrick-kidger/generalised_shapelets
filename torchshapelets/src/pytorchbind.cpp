#include <torch/extension.h>
#include <pybind11/functional.h>

#include "discrepancies.hpp"
#include "shapelet_transform.hpp"

#ifndef _OPENMP
    #error OpenMP required
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2_discrepancy", &torchshapelets::l2_discrepancy);
    m.def("shapelet_transform", &torchshapelets::shapelet_transform);
}