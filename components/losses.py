"""
ADDING A LOSS FUNCTION:
Any loss functions you want to add should be created as a function (or class, see CEDiceLoss) in this file. 
Loss functions, for the purpose of this suite, need to take logits of shape [batch_size, num_classes, height, width], 
masks of shape [batch_size, height, width], should ignore pixels equal to IGNORE_INDEX, and should return a single 
number representing the loss. It needs to be added to the LOSSES registry and have a corresponding <loss_fn_name>.yaml file 
created under conf/loss/
"""

import torch

from components.dataset import IGNORE_INDEX


def cross_entropy_loss():
    """
    Wrapper for the Torch implementation of cross entropy loss, ignoring pixels with value IGNORE_INDEX
    :return:    The Torch cross entropy loss function
    """
    return torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def _dice_loss(logits, masks):
    """
    Measures how much the predicted region for each class overlaps the true region - 0 = perfect, 1 = none
    Result is the average over all classes

    Uses the model's probabilities (softmax) rather than hard predictions (argmax)
    Hard predictions don't change when the weights change slightly, can't direct training

    Pixels equal to IGNORE_INDEX are left out
    The +1 in the formula avoids dividing by zero when class is absent from the batch

    :param logits:  Model output, shape [batch_size, num_classes, height, width]
    :param masks:   Class IDs, shape [batch_size, height, width]
    :return:        A single number, the loss
    """
    total = 0.0
    probabilities = logits.softmax(dim=1)  # 2 scores per pixel -> probabilities that sum to 1
    valid = masks != IGNORE_INDEX  # Ignore pixels that are used as padding
    for cls_id in range(logits.shape[1]):
        predicted = probabilities[:, cls_id][valid]  # Model's probability for this class
        actual = (masks == cls_id)[valid].float()   # 1.0 where the label is this class, 0.0 everywhere else
        overlap = (predicted * actual).sum()
        dice = (2 * overlap + 1) / (predicted.sum() + actual.sum() + 1)
        total += 1 - dice
    return total / logits.shape[1]


class CEDiceLoss:
    """
    Represents the combined loss function: cross-entropy + dice_weight * dice_loss
    Higher dice weight increases importance of dice loss
    Runs using other loss functions shouldn't be directly compared via loss values
    (validation loss is not a consistent measure of model performance when runs use different loss functions)
    Raises an error if a batch has no non-ignored pixels
    """

    def __init__(self, dice_weight: float) -> None:
        self.cross_entropy = cross_entropy_loss()
        self.dice_weight=dice_weight

    def __call__(self, logits, masks):
        """
        Calculates the loss as described by the class docstring
        :param logits:  Model output, shape [batch_size, num_classes, height, width]
        :param masks:   Class IDs, shape [batch_size, height, width]
        :return:        A single number, the loss
        """
        if (masks == IGNORE_INDEX).all():  # No non-ignored pixels
            raise ValueError("Loss requires at least one non-ignored pixel")
        return self.cross_entropy(logits, masks) + self.dice_weight * _dice_loss(logits, masks)

# Registry for loss functions - if you add a loss function, it needs an entry here - <name>: loss_fn, where <name> is the corresponding name in the Hydra config for the loss function.
LOSSES = {
    "ce": cross_entropy_loss,
    "ce_dice": CEDiceLoss,
}

def build_loss(loss_cfg):
    """
    Returns the loss function based on the passed loss_cfg.
    :param loss_cfg:    The loss cfg to be used to build the loss function
    :raises ValueError: Raises ValueError if loss_cfg.name isn't an entry in the LOSSES registry
    :return:            The built loss function
    """

    # Raise an error if we get an invalid loss_cfg
    if loss_cfg.name not in LOSSES:
        raise ValueError(f"{loss_cfg.name} not found in list of valid losses: {sorted(LOSSES)}")

    # Otherwise retrieve the options from the loss_cfg
    options = {key: value for key, value in loss_cfg.items() if key != "name"}

    # Return the built loss function using the unpacked options
    return LOSSES[loss_cfg.name](**options)

