# pyg_data=/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/origin/pyg_data.pt

# python pyg_sage.py \
#     --data=$pyg_data \
#     --epochs=10 \
#     --device=gpu \
#     --log_path=log/pyg_sage.log

dgl_data=/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/origin/dgl_data.bin

python dgl_sage.py \
    --data=$dgl_data \
    --num-epochs=100 \
    --gpu=0 > log/dgl_sage1.log

compressgnn_data=/mnt/disk1/GNNDataset/OGB/dataset/ogbn-products/compress/dgl_data.bin

python compressgnn_sage.py \
    --data=$compressgnn_data \
    --num-epochs=100 \
    --gpu=0 > log/compressgnn_sage1.log

