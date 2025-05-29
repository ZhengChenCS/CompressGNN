# root_dir=/home/chenzheng/CompressGNN/CompressGNNDataset/dataset

# dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000 hollywood-2009)

# for i in {0..0}
# do
#     echo ${dataset[i]}
#     python createCompressDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/compress coo
#     python createCompressDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/compress csr 
# done
root_dir=/mnt/disk1/KongmingDataset
vaild_dataset=(Cora CiteSeer PubMed ogbn_arxiv Reddit AmazonProducts)
for i in {0..2}
do 
    echo ${vaild_dataset[i]}
    # python createCompressDataset.py $root_dir/${vaild_dataset[i]} $root_dir/${vaild_dataset[i]}/compress coo
    python createCompressDataset.py $root_dir/${vaild_dataset[i]} $root_dir/${vaild_dataset[i]}/compress csr 
    # break
done

root_dir=/mnt/disk1/GNNDataset

# dataset=(Reddit AmazonProducts)
# for i in {0..1}
# do
#     echo ${dataset[i]}
#     # python createCompressDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/compress coo
#     python createCompressDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/compress csr 
# done

# root_dir=/mnt/disk1/GNNDataset/OGB/dataset/
# dataset=(ogbn-products)
# for i in {0..0}
# do
#     echo ${dataset[i]}
#     # python createCompressDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/compress coo
#     python createCompressDataset.py $root_dir/${dataset[i]} $root_dir/${dataset[i]}/compress csr 
# done