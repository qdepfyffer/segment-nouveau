"""
ADDING A NEW TRANSFORM:
Transforms can be added here and then imported wherever you're using them. They should accept both an image and a mask,
they need to transform the image and mask in EXACTLY the same way (besides any padding, see example in
training_transform), and they need to return the image and the mask
"""

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

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



class PadToSize:
    """
    Images in test / valid splits should not receive the random cropping, but they might need to be padded (padding goes on the right & on the bottom)
    The sizes of images in our tomato dataset is inconsistent (due to trimming that happened during the original extraction of the RGB images)
    They vary in height (284 ~ 288), this transform lets us ensure that every image in a batch comes out to the same size
    """


    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pads the right and bottom of a pair of tensors which represent a visual image and a mask
        :param image:   The tensor storing the image
        :param mask:    The tensor storing the mask
        :return:        The two tensors, now padded on the right and bottom, as a tuple
        """
        # Calculate the number of pixels needed to pad the image to the desired dimensions
        missing_h = self.height - image.shape[-2]
        missing_w = self.width - image.shape[-1]

        # If either comes out negative our image is too big, we don't want to cut off pixels with this transform
        if missing_h < 0 or missing_w < 0:
            raise ValueError(f"Image is larger than requested pad value: {image.shape[-2]} x {image.shape[-1]} vs {self.height} x {self.width}")

        # Do the actual padding - Masks should be the same size as their images so we pad them the same
        # If you've hit this point, and you have a mask that doesn't match the image size, something else has gone seriously wrong
        image = F.pad(image, [0, 0, missing_w, missing_h], fill=0)  # fill=0 -> Pad visual image with black pixels
        mask = F.pad(mask, [0, 0, missing_w, missing_h], fill=IGNORE_INDEX)  # fill=255 -> Pad mask with "ignored" pixels (using IGNORE_INDEX constant)

        # Return the padded tensors
        return image, mask
