"""
Run various evaluation setups on a model implementation - results will suggest whether the model's size_multiple is set
correctly

If the "as is" run matches / is very close to the "mirror to multiple of 64" run, the model's "size multiple" is likely
correctly set, otherwise raise your size multiple to the next power of 2. 2 -> 4, 4 -> 8, 8 -> 16, etc.

If the score for "black padding of 32 px" is much lower, the model is breaking down when it sees black, this lets you
know whether padding with black will blow up your metrics - but you should avoid it anyway

Run this after the first time you train a new model on each crop you plan to report, you should be able to tell whether
sizing is hurting your metrics
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from components.metrics import scores_from_counts
from dataset import SegmentationDataset
from evaluate import load_checkpoint, count_split
from models.sizing import pad_to_multiple


def parse_args():
    """
    Contains all the command line arguments with reasonable defaults for any that aren't explicitly required.
    :return: Command line arguments to be used when checking models
    """
    parser = argparse.ArgumentParser(
        description="Check the sizing for a given model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the folder containing the desired checkpoint and settings.json")
    parser.add_argument("--crop", type=str, required=True, help="The crop (name of subdirectory within the dataset root directory) being used for checking")
    parser.add_argument("--data_root", type=Path, default=None, help="Optional dataset root override")
    return parser.parse_args()

def pad_to_64(images):
    return pad_to_multiple(images, 64)  # Internal resizing of models commonly uses some power of 2, we use 64 because it's evenly divisible by 32, 16, 8, 4, and 2, so the model should handle it well

def pad_black(images):
    return torch.nn.functional.pad(images, (0, 32, 0, 32))  # Check whether the model loses the plot when it sees black padding

def main():

    args = parse_args()
    model, settings = load_checkpoint(args.checkpoint)

    # Load and build scoring dataset (batch size 1 lets us score every image at native resolution)
    dataset_root = Path(settings["data_root"]) if args.data_root is None else args.data_root
    eval_ds = SegmentationDataset(dataset_root, args.crop, "valid")  # Hard coded to valid split, doesn't really matter which we use but this ensures consistency
    eval_loader = DataLoader(eval_ds, batch_size=1)

    # Dictionary for pad variants
    variants = {
        "as is": None,
        "mirrored to multiple of 64": pad_to_64,
        "black padding of 32 px": pad_black,
    }

    # Run evals
    for name, prepare in variants.items():
        totals = count_split(model, eval_loader, prepare)
        scores = scores_from_counts(**totals)
        print(f"{name:30s} sunlit IoU: {scores['sunlit_iou']:.4f}")

if __name__ == "__main__":
    main()