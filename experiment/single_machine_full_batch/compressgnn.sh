root_dir=/mnt/disk1/GNNDataset
dataset=(Reddit AmazonProducts)
num_features=(602 200)
num_classes=(41 107)
deltaH=(30 5)

# compressgnn
for i in {0..1}
do
    # csr_data=${root_dir}/${dataset[i]}/compress/data_csr_3.pt
    csr_data=${root_dir}/${dataset[i]}/compress/data_csr_8_gorder.pt
    # csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    # python compressgnn_gcn.py \
    # --data=$csr_data \
    # --num_features=${num_features[i]} \
    # --num_classes=${num_classes[i]} \
    # --epochs=10000 \
    # --device=gpu \
    # --deltaH=${deltaH[i]} \
    # --log_path=log/${dataset[i]}_compressgnn_gcn.log

    python compressgnn_sgc.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=10000 \
    --device=gpu \
    --deltaH=${deltaH[i]} \
    --log_path=log/${dataset[i]}_compressgnn_sgc.log
    # break
done
