"""
ADDING A NEW TRANSFORM:
Transforms can be added here as a function and a branch in build_training_transform. They should accept both an image and a mask,
they need to transform the image and mask in EXACTLY the same way, and they need to return the image and the mask
"""

from torchvision.transforms import v2


def rand_crop_transform(height: int, width: int) -> v2.Compose:
    """
    Optional transform for the dataset - crops a random static sized window from the image and randomly mirrors it horizontally w/ p=0.5
    Images smaller than the crop are padded by mirroring before being cropped
    :param height:  Height of the random crop
    :param width:   Width of the random crop
    :return:        Composed transform - random crop and 50% chance to mirror horizontally
    """
    return v2.Compose([
        v2.RandomCrop(size=(height, width), pad_if_needed=True, padding_mode="reflect"),
        v2.RandomHorizontalFlip(p=0.5),
    ])

def stretch_transform(height: int, width: int) -> v2.Compose:  
    """
    Optional transform for the dataset - resizes the image to the specified size and randomly mirrors it horizontally w/ p=0.5
    :param height:  Height to stretch the image to
    :param width:   Width to stretch the image to
    :return:        Composed transform - Stretch the image and 50% chance to mirror horizontally
    """
    return v2.Compose([
        v2.Resize((height, width)),
        v2.RandomHorizontalFlip(p=0.5),
    ])

def build_training_transform(sizing, train_crop):
    """
    Builds the image transform to be applied to images as models are trained.
    :param sizing:      The sizing method to be used
    :param train_crop:  The random crop size to be used in the case of a sizing method / transform that utilizes random cropping
    :raises ValueError: ValueError is raised when a nonexistent sizing method is passed
    :return:            The transform corresponding to the passed sizing
    """
    if sizing.name == "native":
        return rand_crop_transform(*train_crop)
    elif sizing.name == "stretch":
        return stretch_transform(*sizing.size)
    else:
        raise ValueError(f"Unknown sizing: {sizing.name}. Available options: native, stretch")
