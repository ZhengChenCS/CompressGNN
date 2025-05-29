# root_dir=../../../dataset
root_dir=/mnt/disk1/GNNDataset

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000 hollywood-2009 Reddit AmazonProducts)

# for i in {0..5}
# do
#     compress_coo_data=${root_dir}/${dataset[i]}/compress/data_coo_1.pt
#     compress_csr_data=${root_dir}/${dataset[i]}/compress/data_csr_1.pt
#     coo_data=${root_dir}/${dataset[i]}/origin/data_coo.pt
#     csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
#     echo ${dataset[i]}
#     echo "----------compressgnn GAS-----------"
#     python compressgnn_gas.py \
#     --data=${compress_coo_data} \
#     --device=gpu \
#     --run_nums=100
#     python compressgnn_gas.py \
#     --data=${compress_coo_data} \
#     --device=cpu \
#     --run_nums=10

#     echo "----------compressgnn SPMM-----------"
#     python compressgnn_spmm.py \
#     --data=${compress_csr_data} \
#     --device=gpu \
#     --run_nums=100
#     python compressgnn_spmm.py \
#     --data=${compress_csr_data} \
#     --device=cpu \
#     --run_nums=10

#     echo "----------PyG GAS-----------"
#     python pyg_gas.py \
#     --data=${coo_data} \
#     --device=gpu \
#     --run_nums=100
#     python pyg_gas.py \
#     --data=${coo_data} \
#     --device=cpu \
#     --run_nums=10 

#     echo "----------PyG SPMM-----------"
#     python pyg_spmm.py \
#     --data=${csr_data} \
#     --device=gpu \
#     --run_nums=100
#     python pyg_spmm.py \
#     --data=${csr_data} \
#     --device=cpu \
#     --run_nums=10

#     echo "----------DGL SPMM-----------"
#     python dgl_spmm.py \
#     --data=${csr_data} \
#     --device=gpu \
#     --run_nums=100 
#     python dgl_spmm.py \
#     --data=${csr_data} \
#     --device=cpu \
#     --run_nums=10
#     break
# done

for i in {7..7}
do
    # compress_coo_data=${root_dir}/${dataset[i]}/compress/data_coo_3.pt
    # compress_csr_data=${root_dir}/${dataset[i]}/compress/data_csr_3.pt
    compress_csr_data=${root_dir}/${dataset[i]}/compress/data_csr_8_gorder.pt
    # coo_data=${root_dir}/${dataset[i]}/origin/data_coo.pt
    csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    echo ${dataset[i]}

    # echo "----------CompressGNN SPMM-----------"
    # python compressgnn_spmm.py \
    # --data=${compress_csr_data} \
    # --device=gpu \
    # --run_nums=100
    # python compressgnn_spmm.py \
    # --data=${compress_csr_data} \
    # --device=cpu \
    # --run_nums=10


    # echo "----------PyG SPMM-----------"
    # python pyg_spmm.py \
    # --data=${csr_data} \
    # --device=gpu \
    # --run_nums=100
    # python pyg_spmm.py \
    # --data=${csr_data} \
    # --device=cpu \
    # --run_nums=10

    echo "----------DGL SPMM-----------"
    python dgl_spmm.py \
    --data=${csr_data} \
    --device=gpu \
    --run_nums=100 
    # python dgl_spmm.py \
    # --data=${csr_data} \
    # --device=cpu \
    # --run_nums=10

    break
done


