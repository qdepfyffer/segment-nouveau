import torch

from dataset import IGNORE_INDEX

def cross_entropy_loss():
    return torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
