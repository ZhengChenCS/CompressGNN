
dgl_data=/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/origin/dgl_data.bin

# python graphsaint.py \
#     --data=$dgl_data \
#     --num-epochs=10 \
#     --dataset-names=ogbn-products > dgl_saint.log

compressgnn_data=/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/compress/dgl_data.bin

python graphsaint.py \
    --data=$compressgnn_data \
    --num-epochs=2 \
    --dataset-names=ogbn-products-compress > compressgnn_saint.log

