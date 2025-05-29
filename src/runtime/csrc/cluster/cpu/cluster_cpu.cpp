#include "cluster_cpu.h"
#include "../../util.h"

void LSH(const torch::Tensor &input, const torch::Tensor &random_vectors,
         const int32_t param_H, torch::Tensor &inter_tensor,
         ID_DATATYPE &active_bucket);

std::tuple<torch::Tensor, ID_DATATYPE>
cluster_forward_cpu(torch::Tensor input, torch::Tensor random_vectors,
                             const uint32_t param_H) {
    Timer tt;
    AT_ASSERTM(input.dim() == 2, "Input must be a 2-dim tensor");
    // int64_t inputHeight = input.size(0);
    int64_t inputWidth = input.size(1);
    AT_ASSERTM(random_vectors.size(0) == inputWidth,
               "Random vector width must be consistent with input width");
    AT_ASSERTM(param_H < 32, "Paramter H must <= 32");
    Timer timer;
    torch::Tensor inter_tensor;
    torch::Tensor vector_index;
    ID_DATATYPE active_bucket;
    LSH(input, random_vectors, param_H, vector_index, active_bucket);
    timer.toc("LSH");
    return {vector_index, active_bucket};
}

void LSH(const torch::Tensor &input, const torch::Tensor &random_vectors,
         const int32_t param_H, torch::Tensor &vector_index,
         ID_DATATYPE &active_bucket) {
    const int64_t inputHeight = input.size(0);
    const int64_t inputWidth = input.size(1);

    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, input.scalar_type(), "LSH", [&] {
            auto input_ptr = input.data_ptr<scalar_t>();
            int64_t total_buckets = std::pow(2, param_H);
            vector_index =
                torch::zeros(inputHeight, input.options().dtype(torch::kInt));
            auto vector_ids_ptr = vector_index.data_ptr<int32_t>();
            std::vector<int> bucket_flag(total_buckets, 0);
            torch::Tensor hash_vectors = input.mm(random_vectors);
            auto hash_vector_ptr = hash_vectors.data_ptr<scalar_t>();
            const int64_t vector_len = hash_vectors.size(1);
            int64_t i, j;
            for (i = 0; i < inputHeight; i++) {
                ID_DATATYPE id = 0;
                for (j = 0; j < vector_len; j++) {
                    int64_t idx = i * vector_len + j;
                    if (hash_vector_ptr[idx] > 0) {
                        id = (id << 1) | 1;
                    } else {
                        id = id << 1;
                    }
                }
                vector_ids_ptr[i] = id;
                bucket_flag[id] = 1;
            }
            std::vector<ID_DATATYPE> bucket_index(total_buckets, 0);
            bucket_index[0] = bucket_flag[0];
            for (i = 1; i < total_buckets; i++) {
                bucket_index[i] = bucket_index[i - 1] + bucket_flag[i];
            }
            active_bucket = bucket_index[total_buckets - 1];
            for (i = 0; i < inputHeight; i++) {
                ID_DATATYPE id = vector_ids_ptr[i];
                ID_DATATYPE new_id = bucket_index[id];
                vector_ids_ptr[i] = new_id;
            }
        });
}
