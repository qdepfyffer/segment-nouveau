from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# No class_dict.csv - class designations live here (for now at least)
# The classes' position in the list is its class ID - Here, sunlit = 1.
classes = [("Non-sunlit",(165, 42, 42)),
           ("Sunlit leaf",(0, 255, 0))]

# Value of pixels that should be ignored (255 by default) - don't confuse this with the (0, 255, 0) green for sunlit pixels
IGNORE_INDEX = 255


def _load_mask(label_path: Path) -> np.ndarray:
    """
    Returns a numpy array representation of the binary mask pointed to by label_path as 0s and 1s
    :param label_path:  The path to the mask being loaded
    :return:            Numpy array of 0s (Non-sunlit pixels) and 1s (Sunlit pixels)
    """

    # Open the label as an RGB image
    img = np.array(Image.open(label_path).convert("RGB"))

    # Create the mask, where every pixel is initially "unassigned" / "ignored"
    mask = np.full(img.shape[:2], IGNORE_INDEX, dtype=np.uint8)

    # Calculate mask
    for cls_idx, cls in enumerate(classes):
        # For each class, get an array of True/False values based on whether the pixel they represent matches the class's color
        matches = (img == cls[1]).all(axis=-1)
        # Set the value in the mask to that class's index for all matching pixels
        mask[matches] = cls_idx

    # If any "pixels" in the mask still have the value 255, your label image has unexpected colors.
    if (mask == IGNORE_INDEX).any():
        raise RuntimeError(
            f"Mask for {label_path} contains a color that doesn't belong to any class. Please check the label image.")

    # If we didn't error out we can return the mask, it's now a label_h x label_w ndarray of 0s and 1s
    return mask


def _find_pairs(data_root: Path, crop: str, split: str) -> list[tuple[Path, Path]]:
    """
    Returns a sorted list of (visual_img, label_img) tuples for a given crop and split.
    :param data_root:   Root directory containing the dataset, should have a subfolder for each crop - See README.md for expected folder structure
    :param crop:        The crop that the dataset contains (e.g. almond, citrus, grape, pistachio, tomato)
    :param split:       The split that is being searched (test, train, or valid)
    :return:            A list of tuples containing pairs of visual images and their paired labels
    """

    pairs = []

    # This is how the canonical dataset is organized at the moment. If that ever changes, this will need to change too.
    vis_dir = data_root / crop / split / "visual"

    # Validate the visual image directory before we go any further, error out early if we can't find it
    if not vis_dir.is_dir():
        raise FileNotFoundError(
            f"{vis_dir} is not a valid directory. Make sure {data_root} exists and that crop and split arguments are spelled correctly.")

    vis_imgs = sorted(vis_dir.glob("*.jpg"))  # Get sorted list of images from the visual image directory

    # Now we need to find the labels to match the visual images
    # Expected convention for label filenames: *vis_img_stem*_L.png
    for file in vis_imgs:
        stem = file.stem
        lbl_path = data_root / crop / split / "label" / f"{stem}_L.png"
        # Raise an exception if we can't find a label for an image
        if not lbl_path.exists():
            raise FileNotFoundError(
                f"The label for {file} could not be located in {str(data_root / crop / split / 'label')}")
        else:
            # If we haven't errored out we can append the pair (we found the label for the image)
            pairs.append((file, lbl_path))

    # Return the pairs (they're built based on the sorted order of the visual images, so they should be sorted)
    return pairs


# Build the PyTorch Dataset class
class SegmentationDataset(Dataset):

    def __init__(self, data_root: Path, crop: str, split: str, transform=None) -> None:
        """
        Builds the PyTorch Dataset
        :param data_root:   Root directory containing the dataset, should have a subfolder for each crop - See README.md for expected folder structure
        :param crop:        The specific crop being considered (e.g. almond, citrus, grape, pistachio, tomato)
        :param split:       The split that is being considered (test, train, or valid)
        :param transform:   What transform, if any, should be applied to the images and masks - Must accept both image and mask, and return both image and mask
        :return:            None
        """
        # Store the list of pairs found for our dataset
        self.pairs = _find_pairs(data_root, crop, split)
        self.transform = transform

    def __len__(self) -> int:
        """
        :return:    Number of items in the dataset
        """
        return len(self.pairs)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param idx: Index of the item being fetched
        :return:    A tuple of PyTorch tensors representing (visual_image, mask_image) with any optional transforms applied - visual_image: float32, shape=[3, h, w]; mask: int64, shape=[h, w], holds class IDs, 255 = ignored
        """
        # Get the pair from self.pairs at idx and store the paths to the visual and label images
        item = self.pairs[idx]
        vis_path = item[0]
        lbl_path = item[1]

        # Load the visual image and convert it to the format that PyTorch expects
        vis_img = Image.open(vis_path).convert("RGB")
        vis_img = F.to_image(vis_img)
        vis_img = F.to_dtype(vis_img, torch.float32, scale=True)

        # Load the mask and convert it to the format that PyTorch expects
        mask = _load_mask(lbl_path)
        mask = torch.from_numpy(mask).long()
        mask = tv_tensors.Mask(mask)

        # Make sure the images and the labels are the same size. The old segment-main pipeline resized both to 512x512, which hid any potential mismatches
        if mask.shape != vis_img.shape[1:]:
            raise ValueError(f"Mask {lbl_path.stem} of size {mask.shape} does not match image {vis_path.stem} of size {vis_img.shape[1:]}")

        # Apply any optional transforms
        if self.transform is not None:
            vis_img, mask = self.transform(vis_img, mask)

        # Return the pair of (vis_img, mask) as PyTorch Tensors
        return vis_img, mask
