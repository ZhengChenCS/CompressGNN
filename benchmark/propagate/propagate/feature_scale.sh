root=../../../dataset/cnr-2000

feature_dim=(1 2 4 8 16 32 64 128 256 512 1024)

for i in {0..10}
do
    echo "Feature dim : ${feature_dim[i]}"
    compress_coo_data=${root}/compress/feature_scale/coo/${feature_dim[i]}.pt
    compress_csr_data=${root}/compress/feature_scale/csr/${feature_dim[i]}.pt
    coo_data=${root}/origin/feature_scale/coo/${feature_dim[i]}.pt
    csr_data=${root}/origin/feature_scale/csr/${feature_dim[i]}.pt
    echo "----------compressgnn GAS-----------"
    python compressgnn_gas.py \
    --data=${compress_coo_data} \
    --device=gpu \
    --run_nums=100
    python compressgnn_gas.py \
    --data=${compress_coo_data} \
    --device=cpu \
    --run_nums=10

    echo "----------compressgnn SPMM-----------"
    python compressgnn_spmm.py \
    --data=${compress_csr_data} \
    --device=gpu \
    --run_nums=100
    python compressgnn_spmm.py \
    --data=${compress_csr_data} \
    --device=cpu \
    --run_nums=10

    echo "----------PyG GAS-----------"
    python pyg_gas.py \
    --data=${coo_data} \
    --device=gpu \
    --run_nums=100
    python pyg_gas.py \
    --data=${coo_data} \
    --device=cpu \
    --run_nums=10 

    echo "----------PyG SPMM-----------"
    python pyg_spmm.py \
    --data=${csr_data} \
    --device=gpu \
    --run_nums=100
    python pyg_spmm.py \
    --data=${csr_data} \
    --device=cpu \
    --run_nums=10

    echo "----------DGL SPMM-----------"
    python dgl_spmm.py \
    --data=${csr_data} \
    --device=gpu \
    --run_nums=100 
    python dgl_spmm.py \
    --data=${csr_data} \
    --device=cpu \
    --run_nums=10
    break
done
