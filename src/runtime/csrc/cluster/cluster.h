#pragma once
#include "../util.h"
#include <torch/extension.h>

std::tuple<torch::Tensor, ID_DATATYPE>
cluster_forward(torch::Tensor input, torch::Tensor random_vectors,
                         const uint32_t param_h);

// std::vector<torch::Tensor> kongming_cluster_cpu(torch::Tensor input,
// torch::Tensor random_vectors, const int64_t param_H);

// torch::Tensor kongming_reconstruct_cpu(const torch::Tensor& input, const
// torch::Tensor& vector_index);
