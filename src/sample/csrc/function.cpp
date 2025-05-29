#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "neighbor_sample.h"

PYBIND11_MODULE(compressgnn_sample, m) {
    m.def("neighbor_sample_ori_impl", &neighbor_sample_ori_impl, "Neighbor Sample with Normal Graph Format");
    m.def("neighbor_sample_cg_sim", &neighbor_sample_sim_impl, "Neighbor Sample with CompressGraph Format");
}