"""
ADDING A NEW OPTIMIZER:
To add a new optimizer, add its name to the OPTIMIZERS list below and a corresponding branch in the build_optimizer function.
"""

import torch


OPTIMIZERS = ["adamw", "rmsprop"]


def build_optimizer(name: str, parameters, lr: float):
    """
    Builds an optimizer based on a specified name
    :param name:        The name of the optimizer to be built
    :param parameters:  Model parameters
    :param lr:          Learning rate as a float
    :return:            The desired optimizer as specified by name
    """
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    elif name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, alpha=0.995)  # Alpha of 0.995 matches 2026 CIWA paper
    else:
        raise ValueError(f"Optimizer {name} not found in optimizers list. Available optimizers: {sorted(OPTIMIZERS)}")
