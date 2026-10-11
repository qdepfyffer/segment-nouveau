"""
ADDING A NEW MODEL:
If you want to add a model, you'll need to do a few things: Create an implementation file for the model you want to add
and add its name to the MODELS dictionary below. For an example (very simple) implementation file, see deeplabv3.py. In
short, the model's forward function must take a batch of images and return the logits with the shape [batch, classes, height, width].
"""

from models.deeplabv3 import DeepLabV3MobileNet


MODELS = {
    "deeplabv3": DeepLabV3MobileNet
}

def build_model(name: str, num_classes: int):
    """
    Returns the model corresponding to the specified name
    :param name:        The name of the model being created
    :param num_classes: The number of classes the model should be initialized with
    :return:            The built model
    """
    if not name in MODELS:
        raise ValueError(f"Model {name} not found in models list. Available models: {sorted(MODELS)}")

    return MODELS[name](num_classes=num_classes)