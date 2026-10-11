"""
ADDING A NEW TRANSFORM:
Transforms can be added here and then imported wherever you're using them. They should accept both an image and a mask,
they need to transform the image and mask in EXACTLY the same way (besides any padding, see example in
training_transform), and they need to return the image and the mask
"""

from torchvision import tv_tensors
from torchvision.transforms import v2

from dataset import IGNORE_INDEX


def training_transform(height: int, width: int) -> v2.Compose:
    """
    Optional training transform for the dataset - crops a random static sized window from the image and randomly mirrors it horizontally w/ p=0.5
    Images smaller than the crop are padded before being cropped
    :param height:  Height of the random crop
    :param width:   Width of the random crop
    :return:        Composed transform - random crop and 50% chance to mirror horizontally
    """
    return v2.Compose([
        v2.RandomCrop(size=(height, width), pad_if_needed=True, fill={
            tv_tensors.Image: 0,
            tv_tensors.Mask: IGNORE_INDEX,
        }),
        v2.RandomHorizontalFlip(p=0.5),
    ])
