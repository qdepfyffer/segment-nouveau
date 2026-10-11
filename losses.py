import torch

from dataset import IGNORE_INDEX

# Add any loss functions in this file and then import them in the file in which you intend to use them.
# Any function added here takes logits of shape [batch_size, num_classes, height, width] and masks of shape [batch_size, height, width], skips pixels equal to IGNORE_INDEX, and returns a single number

def cross_entropy_loss():
    return torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
