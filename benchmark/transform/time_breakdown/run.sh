root_dir=../../../dataset

dataset=(Cora CiteSeer PubMed Reddit AmazonProducts)
num_features=(1433 3703 500 602 200)
num_classes=(7 6 3 41 102) 
deltaH=(6 6 9 0 0)


for i in {0..4}
do
    data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    python compressgnn_sgc.py \
    --data=${data} \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}
done
