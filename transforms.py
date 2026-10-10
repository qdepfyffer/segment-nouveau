import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


# This is only to change what the model is seeing during training epochs
def training_transform(height: int, width: int):
    """
    Optional training transform for the dataset - crops a random static sized window from the image and randomly mirrors it horizontally w/ p=0.5
    :param height:  Height of the random crop
    :param width:   Width of the random crop
    :return:        Composed transform - random crop and 50% chance to mirror horizontally
    """
    return v2.Compose([
        v2.RandomCrop(size=(height, width), pad_if_needed=True, fill={
            tv_tensors.Image: 0,
            tv_tensors.Mask: 255,
        }),
        v2.RandomHorizontalFlip(p=0.5),
    ])


# Images in test / valid splits should not recieve the random cropping, but for some reason our tomato dataset is not consistent
# They vary in height (284 ~ 288), so we need a way to make sure the model is getting consistently sized images, that is what this transform is for
class PadToSize:

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # Calculate the number of pixels needed to pad the image to the desired dimensions
        missing_h = self.height - image.shape[-2]
        missing_w = self.width - image.shape[-1]

        # If either comes out negative our image is too big, we don't want to cut off pixels with this transform
        if missing_h < 0 or missing_w < 0:
            raise ValueError(f"Image is larger than requested pad value: {image.shape[-2]} x {image.shape[-1]} vs {self.height} x {self.width}")

        # Do the actual padding - Masks should be the same size as their images so we pad them the same
        # If you've hit this point, and you have a mask that doesn't match the image size, something else has gone seriously wrong
        image = F.pad(image, [0, 0, missing_w, missing_h], fill=0)  # fill=0 -> Pad visual image with black pixels
        mask = F.pad(mask, [0, 0, missing_w, missing_h], fill=255)  # fill=255 -> Pad mask with "ignored" pixels

        # Return the padded tensors
        return image, mask
