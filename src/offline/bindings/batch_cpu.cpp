#include "batch_bridge.hpp"
void bind_batch_cpu(pybind11::module_& m) {
    namespace py=pybind11;
    m.def("compress_csr_batch",[](const py::array& v,const py::array& e,int threads,int rounds,int frequency){
        return batch_bridge::run(compressgraph::compress_cpu,v,e,threads,rounds,frequency);
    },py::arg("vlist"),py::arg("elist"),py::arg("threads")=4,py::arg("rounds")=12,py::arg("min_pair_frequency")=16);
}
