import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import math
from torch_geometric.loader import DataLoader
import shutil
import logging
import time
import argparse
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("../../layer/")
sys.path.append("../../loader/")
sys.path.append("../../model")
from CompressgnnData import CompressgnnData
# from sgc_loader import SGCData, SGCDataLoader
from pyg.gcn import GCN



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--num_features', type=int, required=True, help="Input num of features")
parser.add_argument('--num_classes', type=int, required=True, help="Input num of classes")
parser.add_argument('--epochs', type=int, required=True, help="Input num of epoch")
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--log_path', type=str, default="", help="log path")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="gcn_nocache.record", filemode='a') # include timestamp
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps
logging.basicConfig(level=logging.INFO, format='%(message)s', filename=args.log_path, filemode="w")


def train_runtime(model, data, epochs, device, patience=10, min_delta=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    data = data.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)
    model.train()
    y_train = data.y[train_mask].squeeze()
    best_train_loss = float('inf')
    patience_counter = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    for epoch in tqdm(range(epochs)):
        epoch_start = time.perf_counter()
        pred = model(data.x, data.edge_index)
        loss = F.cross_entropy(pred[train_mask], y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_end = time.perf_counter()
        
        epoch_time = epoch_end - epoch_start
        logger.info("Epoch {:d} time: {:.4f}s".format(epoch + 1, epoch_time))
        print("Epoch {:d} time: {:.4f}s".format(epoch + 1, epoch_time))
        
        # 记录训练损失
        logger.info("Epoch {:d} train loss: {:.4f}".format(epoch + 1, loss.item()))
        print("Epoch {:d} train loss: {:.4f}".format(epoch + 1, loss.item()))
        
        # 计算训练集准确率
        model.eval()
        with torch.no_grad():
            test_pred = pred[test_mask].max(dim=1)[1]
            test_correct = test_pred.eq(data.y[test_mask].squeeze()).sum().item()
            test_accuracy = test_correct / test_mask.sum().item() * 100
            logger.info("Epoch {:d} test accuracy: {:.2f}%".format(epoch + 1, test_accuracy))
        
        # 早停机制基于训练损失
        if loss < best_train_loss - min_delta:
            best_train_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info("Early stopping at epoch {:d}".format(epoch + 1))
            break
        
        model.train()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_end - t_start



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = torch.load(args.data)
    logger.info("Dataset=%s", args.data)
    logger.info("Num of features=%s", args.num_features)
    logger.info("Num of classes=%s", args.num_classes)
    dataset.x = dataset.x.to(torch.float32)
    param_H = int(math.log(dataset.x.size()[0], 2))+1

    model = GCN(in_features=args.num_features, out_features=args.num_classes)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    warm_up_time = train_runtime(model, dataset, epochs=5, device=device)
    
    logger.info("Train time for {:d} epoch: {:.4f}s".format(10, warm_up_time))
    

    train_time = train_runtime(model, dataset, epochs=args.epochs, device=device)
    logger.info("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    print("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    