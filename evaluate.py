"""
Script for the evaluation of trained segmentation models
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset
from components.metrics import confusion_counts, scores_from_counts
from models import build_model


def parse_args():
    """
    Contains all the command line arguments used in evaluation with reasonable defaults for any that aren't explicitly required.
    :return: Command line arguments to be used in model evaluation
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a semantic segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the folder containing the desired checkpoint and settings.json")
    parser.add_argument("--crop", type=str, required=True, help="The crop (name of subdirectory within the dataset root directory) you want to use in evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "valid"], help="The split the model should consider for evaluation")
    parser.add_argument("--output_dir", type=Path, default=Path("eval"), help="The folder this script should output to")
    parser.add_argument("--data_root", type=Path, default=None, help="Optional dataset root override")
    return parser.parse_args()

def main():
    # Grab new command line arguments
    args = parse_args()

    # Load saved settings
    settings_path = args.checkpoint / "settings.json"
    with open(settings_path) as file:
        settings = json.load(file)

    # Determine output directory
    out_dir = args.output_dir / args.checkpoint.name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / f"{args.crop}-{args.split}.json"

    # Get path to checkpoints & raise an error if they can't be found
    checkpoint_path = args.checkpoint / f"{settings['architecture']}-best.pt"
    if not checkpoint_path.is_file():
        # We should only end up here if the checkpoints are missing but the settings aren't, which shouldn't ever happen
        raise FileNotFoundError(f"Model checkpoint could not be found at {checkpoint_path}. Please double check and try again.")

    # Build the model based on saved settings, move it to GPU, and put it in evaluation mode
    model = build_model(settings["architecture"], num_classes=2)  # CIWA+ only uses 2 classes
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))  # We only load the weights for safety
    model.to("cuda")
    model.eval()

    # Load and build scoring dataset (batch size 1 lets us score every image at native resolution)
    dataset_root = Path(settings["data_root"]) if args.data_root is None else args.data_root
    eval_ds = SegmentationDataset(dataset_root, args.crop, args.split)
    eval_loader = DataLoader(eval_ds, batch_size=1)

    # Dictionary to store running totals
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    # Begin evaluation
    with torch.no_grad():
        for images, masks in tqdm(eval_loader, desc=f"[EVAL]  {args.split}"):
            # Put images and masks on GPU
            images = images.to("cuda")
            masks = masks.to("cuda")

            # Run the model and get concrete predictions
            logits = model(images)
            predictions = logits.argmax(dim=1)

            # Get confusion counts, add to running total of each
            current_conf = confusion_counts(predictions, masks)
            for key in totals:
                totals[key] += current_conf[key]

    # Score the model once with totals
    # By default models are scored at native label resolution with no crop or resize
    scores = scores_from_counts(**totals)  # Unpacks the values from totals as arguments

    tqdm.write(f"[INFO]  Scored across all {len(eval_ds)} images in the {args.crop} {args.split} split")
    tqdm.write(f"[EVAL]  Counts: TP: {totals['tp']}; FP: {totals['fp']}; FN: {totals['fn']}; TN: {totals['tn']};")
    for key in scores:
        tqdm.write(f"[EVAL]  {key}: {scores[key]:.5f}")

    tqdm.write(f"[INFO]  Saving metrics to {eval_path}")

    # Build metrics dictionary with all desired information
    eval_metrics = {
        "model": settings,
        "evaluation": {
            "data_root": str(dataset_root),
            "crop": args.crop,
            "split": args.split,
            "images": len(eval_ds),
            "pixels": sum(totals.values())
        },
        "metrics": {
            "counts": totals,
            "scores": scores,
        }
    }

    # Write metrics to file
    with open(eval_path, "w") as file:
        json.dump(eval_metrics, file, indent=4)


if __name__ == "__main__":
    main()