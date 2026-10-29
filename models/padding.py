"""
No touchy unless you have a very good reason. Contains a function to safely pad image batches via reflecting pixels
instead of padding with black pixels, which the models apparently do not like. We pad because the model shrinks images
by a fixed factor, and if the image size doesn't divide cleanly its output can land a few pixels off.
"""

import torch
import torch.nn.functional

def pad_to_multiple(images: torch.Tensor, multiple: int) -> torch.Tensor:
    """
    Safely pads a batch of images on the right/bottom via reflection so height and width are a clean multiple of `multiple`
    :param images:      The batch of images being padded
    :param multiple:    The multiple which images should be evenly divisible by
    :return:            The padded images
    """
    height, width = images.shape[-2:]
    missing_h = (-height) % multiple
    missing_w = (-width) % multiple
    return torch.nn.functional.pad(images, (0, missing_w, 0, missing_h), mode="reflect")