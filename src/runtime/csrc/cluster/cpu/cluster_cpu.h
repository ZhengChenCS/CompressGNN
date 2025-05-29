#pragma once
#include "../../util.h"
#include <torch/extension.h>

std::tuple<torch::Tensor, ID_DATATYPE>
cluster_forward_cpu(torch::Tensor input, torch::Tensor random_vectors,
                             const uint32_t param_H);