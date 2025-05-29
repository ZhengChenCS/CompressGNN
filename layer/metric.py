import torch
from torch import Tensor


def cosine_distance(A: Tensor):
    prod = torch.mm(A, A.t())
    norm = torch.norm(A, p=2, dim=1).unsqueeze(0)
    norm.clamp_(1e-6)
    cos = prod.div(torch.mm(norm.t(), norm))
    tri = torch.triu(cos, diagonal=1)
    factor = 2 * A.size()[0] / (A.size()[0] - 1)
    return torch.mean(tri) * factor


def manhattan_distance(A: Tensor):
    pass


def t2t_distance(A: Tensor, B: Tensor, tri: bool = False):
    prod = torch.mm(A, B.t())
    normA = torch.norm(A, p=2, dim=1).unsqueeze(0)
    normA.clamp_(1e-6)
    normB = torch.norm(B, p=2, dim=1).unsqueeze(0)
    normB.clamp_(1e-6)
    cos = prod.div(torch.mm(normA.t(), normB))
    if tri == True:
        cos = torch.triu(cos, diagonal=1)
    return torch.sum(cos)


def batch_distance(A: Tensor, batch_size=1024):
    distance = 0
    row = A.size()[0]
    num_batch = (int)((row + batch_size - 1) / batch_size)
    for batch in range(num_batch):
        start = batch * batch_size
        if batch == num_batch - 1:
            end = row
        else:
            end = (batch + 1) * batch_size
        left = A[start:end]
        distance += t2t_distance(left, left, tri=True)
        for comp in range(batch):
            right_start = comp * batch_size
            right_end = (comp + 1) * batch_size
            right = A[right_start:right_end]
            distance += t2t_distance(left, right)
    factor = row * (row - 1) / 2
    return distance / factor
