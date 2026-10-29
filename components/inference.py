"""
Contains implementations related to running inference with models and getting the models' predictions
If you add a new sizing method / transform you also need to add a branch here
"""

import torch.nn.functional
from torchvision.transforms.v2 import functional as TF


def predict_logits(model, images, sizing):
    """
    Runs inference with a given model across a set of images
    :param model:   The model to be used
    :param images:  The images to generate predictions from
    :param sizing:  The sizing method used
    :raises ValueError: ValueError is raised when a nonexistent sizing method is passed
    :return:        The logits as output by the model
    """

    if sizing.name == "native":
        return model(images)  # Return the logits as output by the model, native sizing has the model pad and crop internally so we don't need to do anything special here
    elif sizing.name == "stretch":
        height, width = images.shape[-2:]  # Get the native image size
        stretched = TF.resize(images, list(sizing.size), antialias=True)  # Stretch the images as they're stretched by the training transform (this is functionally the same as the v2.Resize in transforms.py)
        logits = model(stretched)  # Get predictions based on the stretched images (this is how the tomato paper's pipeline worked)
        out = torch.nn.functional.interpolate(logits, (height, width), mode="bilinear", align_corners=False)  # Resize the logits back to the images' native sizes
        return out
    else:
        raise ValueError(f"Unknown sizing: {sizing.name}. Available options: native, stretch")