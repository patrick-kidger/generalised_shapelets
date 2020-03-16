#ifndef TORCHSHAPELETS_PYCAPSULE_HPP
#define TORCHSHAPELETS_PYCAPSULE_HPP

#include <torch/extension.h>


namespace torchshapelets {
    // Makes an instance of a struct of type T and wraps it into a PyCapsule.
    template <typename T, typename ...Args>
    inline py::object wrap_capsule(Args&&... args);

    // Unwraps a capsule to give a struct of type T
    template <typename T>
    inline T* unwrap_capsule(py::object capsule);
}  // namespace torchshapelets

#include "pycapsule.inl"

#endif //TORCHSHAPELETS_PYCAPSULE_HPP