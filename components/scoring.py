"""
Contains implementations pertaining to the loading, evaluation, and scoring of models
"""

import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.metrics import confusion_counts, scores_from_counts
from dataset import SegmentationDataset
from models import build_model

# Set up logging
log = logging.getLogger(__name__)

# The below functions were pulled out so they could be imported and reused elsewhere because they implement patterns that get repeated in other files
def load_checkpoint(checkpoint_dir: Path, device):
    """
    Reads config.yaml from a checkpoint's folder and rebuilds the model by loading the saved weights and moves the model to the specified device
    :param checkpoint_dir:  Directory holding the checkpoint and the config.yaml that resulted in that set of weights
    :param device:          The device the model should be moved to (cuda, cuda:n, cpu)
    :return:                The rebuilt model and its training config
    """
    # Load saved settings
    config_path = checkpoint_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"'config.yaml' not found at {config_path}.")
    settings = OmegaConf.load(config_path)

    # Get path to checkpoints & raise an error if they can't be found
    checkpoint_path = checkpoint_dir / f"{settings.model.name}-best.pt"
    if not checkpoint_path.is_file():
        # We should only end up here if the checkpoints are missing but the settings aren't, which shouldn't ever happen
        raise FileNotFoundError(f"Model checkpoint could not be found at {checkpoint_path}. Please double check and try again.")

    # Build the model based on saved settings, move it to specified device
    model = build_model(settings.model.name, num_classes=2)  # CIWA+ only uses 2 classes
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))  # We only load the weights for safety
    model.to(device)
    model.eval()

    return model, settings


def count_split(model, loader, prepare=None):
    """
    Runs the model on every image in the loader and returns summed tp/fp/fn/tn counts
    :param model:   The model to be run
    :param loader:  The DataLoader that will feed the model
    :param prepare: Optional function that gets applied to each image batch before being passed to the model (leave as None if you aren't absolutely sure you want something here)
    :return:        A dictionary holding summed tp/fp/fn/tn counts
    """
    # Dictionary to store running totals
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    # Send batches to the same device as the model
    device = next(model.parameters()).device  # model.parameters() -> weights, next() gives the first one, .device says where it is

    # Begin evaluation
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="[EVAL]  Progress"):
            # Put images and masks on the same device as the model
            images = images.to(device)
            masks = masks.to(device)
            height, width = masks.shape[-2:]

            if prepare is not None:
                images = prepare(images)

            # Run the model and get concrete predictions
            predictions = model(images).argmax(dim=1)[..., :height, :width]

            # Get confusion counts, add to running total of each - see components/metrics.py for implementation
            current_conf = confusion_counts(predictions, masks)
            for key in totals:
                totals[key] += current_conf[key]

    return totals


def score_run(model, settings, data_root, crop, split, out_dir):
    """
    Scores a model with the given settings over the dataset specified by data_root, crop, and split - results saved to out_dir
    :param model:       The model to be evaluated
    :param settings:    The model's training config as returned by load_checkpoint()
    :param data_root:   The root directory holding the datasets
    :param crop:        Which crop is being used in evaluation  
    :param split:       Which split the model is being evaluated against 
    :param out_dir:     The output directory for model metrics
    :return:            The metrics dict that was written to metrics.json
    """
    # Make sure out_dir exists and set output path for evaluation
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "metrics.json"

    # Load and build scoring dataset (batch size 1 lets us score every image at native resolution)
    dataset_root = Path(data_root) if data_root is not None else Path(settings.data_root)
    eval_ds = SegmentationDataset(dataset_root, crop, split)
    eval_loader = DataLoader(eval_ds, batch_size=1)

    # Calculate total tp/fp/fn/tn counts
    totals = count_split(model, eval_loader)

    # Score the model once with totals
    # By default models are scored at native label resolution with no crop or resize
    scores = scores_from_counts(**totals)  # Unpacks the values from totals as arguments

    log.info(f"Scored across all {len(eval_ds)} images in the {crop} {split} split")
    log.info(f"Counts: TP: {totals['tp']}; FP: {totals['fp']}; FN: {totals['fn']}; TN: {totals['tn']};")
    for key in scores:
        log.info(f"{key}: {scores[key]:.5f}")

    log.info(f"Saving metrics to {eval_path}")

    # Build metrics dictionary with all desired information
    eval_metrics = {
        "model": OmegaConf.to_container(settings),
        "evaluation": {
            "data_root": str(dataset_root),
            "crop": crop,
            "split": split,
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


    # Return eval_metrics (everything can be retrieved from this)
    return eval_metrics