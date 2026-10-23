"""
Script for the training of segmentation models
"""

import json
import logging
import math
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.losses import build_loss
from components.metrics import confusion_counts, scores_from_counts
from components.optimizers import build_optimizer
from components.transforms import training_transform
from dataset import SegmentationDataset
from evaluate import load_checkpoint, score_run
from models import build_model

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    This is where the training loop lives.
    :return: None
    """

    # Make sure a valid monitor argument has been passed
    MONITORS = ["valid_loss", "sunlit_iou", "f1", "mean_iou", "precision"]
    if cfg.monitor not in MONITORS:
        raise ValueError(f"{cfg.monitor} not found in list of valid monitors: " + " ".join(MONITORS))

    # Make sure cfg.eval_splits is parsed correctly
    eval_splits = cfg.eval_splits if cfg.eval_splits is not None else []
    if isinstance(eval_splits, str):
        eval_splits = [eval_splits]

    # Raise a ValueError if we get a split that doesn't really exist 
    AVAILABLE_SPLITS = ["test", "train", "valid"]
    for split in eval_splits:
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"{split} not found in list of valid splits: " + " ".join(AVAILABLE_SPLITS))

    # Make sure we get an absolute path for our data_root so scripts always look in the exact same place
    if not Path(cfg.data_root).expanduser().is_absolute():
        raise ValueError(f"{cfg.data_root} is not an absolute path - please use absolute paths for the data_root argument to ensure consistency")


    # Try to put any important checks above here so this script exits before it eats up your resources
    

    # Use our RNG seed
    torch.manual_seed(cfg.seed)

    # Determine output directory
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    best_model_path = out_dir / f"{cfg.model.name}-best.pt"

    # Build specified model
    model = build_model(cfg.model.name, num_classes=2)
    model = model.to("cuda")  # Move model to GPU

    # Load our datasets
    train_ds = SegmentationDataset(Path(cfg.data_root), cfg.crop.name, "train",
                                   transform=training_transform(cfg.crop.train_crop[0], cfg.crop.train_crop[1]))
    valid_ds = SegmentationDataset(Path(cfg.data_root), cfg.crop.name, "valid")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=1)  # Validate with a batch size of 1 so images don't need to be the same size

    # Set optimizer and loss function
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    loss_fn = build_loss(cfg.dice_weight)

    # Initialize best_value to negative infinity, we use a higher-is-better approach for monitoring model performance
    # Loss will need to be negated during the training for this calculation
    best_value = float("-inf")

    # Initialize history as an empty list so we can track the model's scores and plot training curves later
    history = []

    # Begin model training
    epochs_without_improvement = 0
    model.train()
    for epoch in range(cfg.epochs):
        # Training loop
        total_train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch + 1}"):

            # Put images and masks on GPU
            images = images.to("cuda")
            masks = masks.to("cuda")

            logits = model(images)
            optimizer.zero_grad()
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        log.info(f"[TRAIN] Loss: {train_loss:.4f}")

        # I didn't even know this was really a thing but if we hit NaN or infinity for a loss value, the model is cooked so we need to just exit
        # Otherwise we would just uselessly train until patience runs out, which again is pointless because we know the model won't improve from here
        if not math.isfinite(train_loss):
            raise RuntimeError(f"[TRAIN] Epoch {epoch + 1}: Loss is out of acceptable bounds, model will not recover - try again with a lower learning rate.")

        # Dictionary to store running totals
        totals = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

        # Validation loop
        total_valid_loss = 0.0

        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"[VALID] Epoch {epoch + 1}"):

                # Put images and masks on GPU
                images = images.to("cuda")
                masks = masks.to("cuda")

                # Get model's predictions and calculate loss
                logits = model(images)
                predictions = logits.argmax(dim=1)
                loss = loss_fn(logits, masks)
                total_valid_loss += loss.item()

                # Get confusion counts, add to running total of each
                current_conf = confusion_counts(predictions, masks)
                for key in totals:
                    totals[key] += current_conf[key]

        # Calculate scores for the current run and store the results
        valid_loss = total_valid_loss / len(valid_loader)
        scores = scores_from_counts(**totals)  # Unpacks the values from totals as arguments
        results = {"valid_loss": valid_loss, **scores}

        # Append our results to the history and write our history to the history file
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **results,
        })
        with open(out_dir / "history.json", "w") as file:
            json.dump(history, file, indent=4)

        # Get the current value for our monitored metric and invert it if we're monitoring loss (we need higher = better)
        current_value = results[cfg.monitor]
        if cfg.monitor == "valid_loss":
            current_value = -current_value

        log.info(f"[VALID] Loss: {valid_loss:.4f}")
        log.info(f"[VALID] Sunlit IOU: {scores['sunlit_iou']:.4f}\tMean IOU: {scores['mean_iou']:.4f}")
        log.info(f"[VALID] Precision: {scores['precision']:.4f}\tAccuracy: {scores['accuracy']:.4f}")
        log.info(f"[VALID] Recall: {scores['recall']:.4f}\t\tF1: {scores['f1']:.4f}")

        # Check if we've gotten a better score than our current best for the monitored value
        if current_value > best_value:
            # If we have a better score, update our current best, save the current weights, inform the user, and reset our patience counter
            best_value = current_value

            # Prevent anything from reading a half-written set of weights (since torch takes a second to write the file)
            temp_path = best_model_path.with_suffix(".tmp")  # Temporary file next to the real one - with_suffix() creates a path only; no file exists
            torch.save(model.state_dict(), temp_path)  # Write the weights to the temporary file - the actual weights file that is read by scripts in this repo isn't touched yet
            temp_path.replace(best_model_path)  # Then swap it in (the finished weights file) - rename the .tmp file to .pt, replacing the old file (.replace() is used instead of .rename() because .rename() fails on windows for existing paths)

            log.info(f"Best model saved to: {best_model_path}")
            epochs_without_improvement = 0
        else:
            # Otherwise don't save the model, increment our patience counter and inform the user that the model hasn't improved
            epochs_without_improvement += 1
            log.info(f"Early stopping: {epochs_without_improvement} / {cfg.patience} epochs without improvement of {cfg.monitor}")

        # Break out of the epoch loop if we hit our patience limit
        if epochs_without_improvement >= cfg.patience:
            log.info(f"{cfg.monitor} has not improved in {cfg.patience} epochs - training stopped")
            break

        # Put the model back in training mode before the next iteration
        model.train()

    del model, optimizer  # Drop the training copy & everything the optimizer holds so it doesn't sit in memory alongside the loaded best checkpoint for evaluation
    torch.cuda.empty_cache()  # Hand back freed memory

    if not eval_splits:
        return  # Exit here if we're not running auto evaluation of the final trained model (no eval_split specified by conf/train.yaml or user override in CLI)

    # Otherwise we try to load the best model and evaluate it over it's OWN dataset (evaluate.py can still evaluate a model on different datasets)
    try:
        if not best_model_path.exists():
            log.warning(f"Checkpoint not found: {best_model_path}")  # Exit early if the checkpoint doesn't exist for any reason
            return
        model, settings = load_checkpoint(out_dir)
        for split in eval_splits:
            score_run(model, settings, None, cfg.crop.name, split, out_dir / "eval" / f"{cfg.crop.name}-{split}")
    except Exception:
        log.exception("Automatic scoring failed - training results intact, score them with evaluate.py")

if __name__ == "__main__":
    main()
