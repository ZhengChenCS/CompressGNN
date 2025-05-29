root_dir=/mnt/disk1/GNNDataset
dataset=(Reddit AmazonProducts)
num_features=(602 200)
num_classes=(41 107)

for i in {0..1}
do
    csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    log_path=${dataset[i]}_dgl
    # python dgl_sgc.py \
    #  --data=$csr_data \
    #  --num_features=${num_features[i]} \
    #  --num_classes=${num_classes[i]} \
    #  --epochs=10000 \
    #  --device=gpu \
    #  --log_path=log/${log_path}_sgc.log
    python dgl_gcn.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=10000 \
    --device=gpu \
    --log_path=log/${log_path}_gcn.log
    break
done
