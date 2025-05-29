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
sys.path.append("../../model/compressgnn")
from CompressgnnData import CompressgnnData
# from sgc_loader import SGCData, SGCDataLoader
from sgc import SGC



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--num_features', type=int, required=True, help="Input num of features")
parser.add_argument('--num_classes', type=int, required=True, help="Input num of classes")
parser.add_argument('--epochs', type=int, required=True, help="Input num of epoch")
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--deltaH', type=int, default=0, help="delta H")
parser.add_argument('--log_path', type=str, default="", help="log path")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="gcn_nocache.record", filemode='a') # include timestamp
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps
logging.basicConfig(level=logging.INFO, format='%(message)s', filename=args.log_path, filemode='w')

def train_runtime(model, data, epochs, device, patience=10, min_delta=0.001, min_rate=0.00001, min_epoch=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    data = data.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)
    model.train()
    y = data.y[train_mask].squeeze()
    best_train_loss = float('inf')
    patience_counter = 0
    is_cluster = True
    loss_history = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for epoch in tqdm(range(epochs)):
        epoch_start = time.perf_counter()
        if data.graph_type == "coo":
            pred = model(data.x, data.edge_index, is_cluster, data.edge_weight, data.vertex_cnt, data.rule_cnt)
        else:
            pred = model(data.x, data.edge_index, is_cluster)
        loss = F.cross_entropy(pred[train_mask], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
            # 使用测试集掩码
            test_pred = pred[test_mask].max(dim=1)[1]
            test_correct = test_pred.eq(data.y[test_mask].squeeze()).sum().item()
            test_accuracy = test_correct / test_mask.sum().item() * 100
            logger.info("Epoch {:d} test accuracy: {:.2f}%".format(epoch + 1, test_accuracy))
        
        if is_cluster == True:
            loss_history.append(loss.item())
            if len(loss_history) > min_epoch:
                loss_rate = abs(loss_history[-1] - loss_history[-2]) / loss_history[-2]
                if loss_rate < min_rate:
                    is_cluster = False
                    logger.info("Switch to no cluster at epoch {:d}".format(epoch + 1))
        else:
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
    param_H = int(math.log(dataset.x.size()[0], 2))+args.deltaH
    param_H = param_H if param_H < 30 else 30

    model = SGC(in_features=args.num_features, out_features=args.num_classes, param_H=param_H, K=2)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    warm_up_time = train_runtime(model, dataset, epochs=10, device=device)
    
    logger.info("Train time for {:d} epoch: {:.4f}s".format(10, warm_up_time))
    
    model.reset_cache()

    train_time = train_runtime(model, dataset, epochs=args.epochs, device=device)
    logger.info("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    # print("Propagate time:{:.4f}".format(model.p_time))
    print("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    