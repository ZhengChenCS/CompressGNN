#pragma once
#include "../../util.h"
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> cluster_workspace_cuda(torch::Tensor input, int64_t bits);
std::tuple<torch::Tensor, torch::Tensor, int64_t> cluster_assign_cuda(
    torch::Tensor input, torch::Tensor planes, int64_t bits, std::vector<torch::Tensor> workspace);
torch::Tensor cluster_mean_cuda(torch::Tensor input, torch::Tensor index, torch::Tensor count);
torch::Tensor cluster_mean_backward_cuda(torch::Tensor grad, torch::Tensor index, torch::Tensor count);
torch::Tensor cluster_reconstruct_cuda(torch::Tensor reps, torch::Tensor index,
    torch::optional<torch::Tensor> bias, bool relu);
torch::Tensor cluster_reconstruct_backward_cuda(torch::Tensor grad, torch::Tensor index,
    torch::Tensor output, int64_t groups, bool relu);
