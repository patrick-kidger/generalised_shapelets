#include <torch/extension.h>  // to get the pybind11 stuff

#include "shapelet_transform.cpp"

#ifndef _OPENMP
    #error OpenMP required
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assert_increasing", &torchshapelets::detail::assert_increasing);
    m.def("len_index", &torchshapelets::detail::len_index);
    m.def("restriction", &torchshapelets::detail::restriction);
    m.def("add_knots", &torchshapelets::detail::add_knots);
    m.def("shapelet_transform", &torchshapelets::shapelet_transform);
}