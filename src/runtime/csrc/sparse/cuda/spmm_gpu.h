#pragma once
#include "../../util.h"
#include <torch/extension.h>


torch::Tensor spmm_cuda(torch::Tensor vlist, torch::Tensor elist,
                        torch::Tensor value, torch::Tensor mat,
                        std::string method,
                        torch::optional<torch::Tensor> optional_out);
