#include "spmm_cpu.h"
#include "../../util.h"
#include <ATen/Parallel.h>
#include <omp.h>
#include "cpu_util.h"

torch::Tensor spmm_cpu(torch::Tensor vlist, torch::Tensor elist,
                       torch::Tensor value, torch::Tensor mat,
                       std::string method, 
                       torch::optional<torch::Tensor> optional_out) {
    CHECK_CPU(vlist);
    CHECK_CPU(elist);
    CHECK_CPU(value);
    CHECK_CPU(mat);
    CHECK_CONTIGUOUS(mat);
    if(optional_out.has_value())
        CHECK_CPU(optional_out.value());
    
    torch::Tensor out;
    if(optional_out.has_value())
    {
        out = optional_out.value().contiguous();
    }
    else{
        auto sizes = mat.sizes().vec();
        sizes[0] = vlist.size(-1)-1;
        out = torch::zeros(sizes, mat.options());
    }
    
    AT_DISPATCH_FLOATING_TYPES(
        mat.scalar_type(), "_", ([&] {
            auto mat_data = mat.data_ptr<scalar_t>();
            auto vlist_ptr = vlist.data_ptr<int64_t>();
            auto elist_ptr = elist.data_ptr<int64_t>();
            auto value_ptr = value.data_ptr<scalar_t>();
            auto out_data = out.data_ptr<scalar_t>();

            auto col_num = mat.size(-1);
            auto row_num = vlist.size(-1) - 1;
            auto nnz = elist.size(-1);
            
            if(method == "rowbalance"){
                int64_t grain_size =
                    at::internal::GRAIN_SIZE /
                    (col_num * std::max(nnz / row_num, (int64_t)1));
                at::parallel_for(
                    0, row_num, grain_size, [&](int64_t begin, int64_t end) {
                        scalar_t val;
                        std::vector<scalar_t> vals(col_num, 0);
                        int64_t row_start, row_end, dst, vid, eid;
                        for (auto i = begin; i < end; i++) {
                            vid = i;
                            row_start = vlist_ptr[vid];
                            row_end = vlist_ptr[vid + 1];
                            for (auto k = 0; k < col_num; k++) {
                                vals[k] = 0;
                            }
                            for (eid = row_start; eid < row_end; eid++) {
                                val = value_ptr[eid];
                                dst = elist_ptr[eid];
                                for (auto k = 0; k < col_num; k++) {
                                    vals[k] += mat_data[dst * col_num + k] * val;
                                }
                            }
                            for (auto k = 0; k < col_num; k++) {
                                out_data[vid * col_num + k] += vals[k];
                            }
                        }
                    });
            }
            else if(method == "nnzbalance"){
                int64_t grain_size = at::internal::GRAIN_SIZE / (col_num * NE_PER_THREAD);
                const int64_t num_block = (nnz + NE_PER_THREAD - 1) / NE_PER_THREAD;
                at::parallel_for(
                    0, num_block, grain_size, [&](int64_t begin, int64_t end){
                        scalar_t val;
                        std::vector<scalar_t> vals(col_num, 0);
                        for(auto i = begin; i < end; i++){
                            int64_t nnz_start = i * NE_PER_THREAD;
                            int eid = nnz_start;
                            int64_t row = binary_search_segment_number<int64_t>(vlist_ptr, row_num, nnz, eid);
                            int step = vlist_ptr[row+1] - eid;
                            int64_t dst;
                            for(int ii = 0; ii < col_num; ii++){
                                vals[ii] = 0;
                            }
                            for(int ii = 0; ii < NE_PER_THREAD; ii++){
                                if(eid > nnz)
                                    break;
                                if(ii < step){
                                    dst = elist_ptr[eid] * col_num;
                                    val = value_ptr[eid];
                                    for(auto k = 0; k < col_num; k++){
                                        vals[k] += mat_data[dst+k] * val;
                                    }
                                    eid++;
                                } else {
                                    for(auto k = 0; k < col_num; k++){
                                        write_add<scalar_t>(&out_data[row*col_num+k], vals[k]);
                                    }
                                    row = binary_search_segment_number<int64_t>(vlist_ptr, row_num, nnz, eid);
                                    step = vlist_ptr[row+1] - eid + ii;
                                    dst = elist_ptr[eid] * col_num;
                                    val = value_ptr[eid];
                                    for(auto k = 0; k < col_num; k++){
                                        vals[k] = mat_data[dst+k] * val;
                                    }
                                    eid++;
                                }
                            }
                            for(auto k = 0; k < col_num; k++){
                                write_add<scalar_t>(&out_data[row*col_num+k], vals[k]); 
                            }
                        }
                });
            }
            else{
                std::cerr << "Unvaild spmm method for CPU." << std::endl; 
            }
        }));
    return out;
}