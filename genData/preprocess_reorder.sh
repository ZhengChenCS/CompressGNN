root_dir=/mnt/disk1/GNNDataset

dataset=(Reddit AmazonProducts)
for i in {0..0}
do
    echo ${dataset[i]}
    # python createCompressDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/compress coo
    python createReorderDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/origin csr 
    # python check_reorder.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/origin csr 
done
