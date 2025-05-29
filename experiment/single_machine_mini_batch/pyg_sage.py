import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import math
from torch_geometric.loader import NeighborLoader
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
from pyg.sage import SAGE
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Input graph data")
parser.add_argument('--epochs', type=int, required=True, help="Input num of epoch")
parser.add_argument('--device', type=str, default='cpu', help="device")
parser.add_argument('--log_path', type=str, default="", help="log path")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename="gcn_nocache.record", filemode='a') # include timestamp
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamps
logging.basicConfig(level=logging.INFO, format='%(message)s', filename=args.log_path, filemode="w")


@torch.no_grad()
def test(data, model, subgraph_loader, device):
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader, device).argmax(dim=-1)
    y = data.y.to(y_hat.device)
    acc = (y_hat[data.test_mask] == y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc


def train_runtime(model, data, epochs, device, patience=3, min_delta=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    # data = data.to(device)
    train_idx = data.train_mask.nonzero().squeeze().to(device)
    val_idx = data.val_mask.nonzero().squeeze().to(device)
    test_idx = data.test_mask.nonzero().squeeze().to(device)

    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[20, 20, 20],
        batch_size=256,
        num_workers=12,
        shuffle=True
    )

    subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    best_train_loss = float('inf')
    patience_counter = 0

    t_start = time.perf_counter()

    

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze().to(torch.long)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_end = time.perf_counter()

        epoch_time = epoch_end - epoch_start
        logger.info("Epoch {:d} time: {:.4f}s".format(epoch + 1, epoch_time))
        print("Epoch {:d} time: {:.4f}s".format(epoch + 1, epoch_time))

        accu = test(data, model, subgraph_loader, device)
        logger.info("Epoch {:d} test accuracy: {:.2f}%".format(epoch + 1, accu))

        avg_loss = total_loss / num_batches

        # 早停检查 (基于loss)
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch}')
            logger.info(f'Best loss: {best_loss:.4f}')
            break

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = torch.load(args.data)
    logger.info("Dataset=%s", args.data)
    dataset.x = dataset.x.to(torch.float32)
    num_features = 100
    num_classes = 47

    model = SAGE(num_features, 128, num_classes, 3)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # warm_up_time = train_runtime(model, dataset, epochs=1, device=device)
    
    # logger.info("Train time for {:d} epoch: {:.4f}s".format(10, warm_up_time))
    

    train_time = train_runtime(model, dataset, epochs=args.epochs, device=device)
    logger.info("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))
    print("Train time for {:d} epoch: {:.4f}s".format(args.epochs, train_time))

