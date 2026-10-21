"""
Script for the training of segmentation models
"""

import json
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.losses import build_loss
from components.optimizers import build_optimizer
from components.transforms import training_transform
from dataset import SegmentationDataset
from models import build_model
from components.metrics import confusion_counts, scores_from_counts


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
        tqdm.write(f"[TRAIN] Loss: {train_loss:.4f}")

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

        tqdm.write(f"[VALID] Loss: {valid_loss:.4f}")
        tqdm.write(f"[VALID] Sunlit IOU: {scores['sunlit_iou']:.4f}\tMean IOU: {scores['mean_iou']:.4f}")
        tqdm.write(f"[VALID] Precision: {scores['precision']:.4f}\tAccuracy: {scores['accuracy']:.4f}")
        tqdm.write(f"[VALID] Recall: {scores['recall']:.4f}\t\tF1: {scores['f1']:.4f}")

        # Check if we've gotten a better score than our current best for the monitored value
        if current_value > best_value:
            # If we have a better score, update our current best, save the current weights, inform the user, and reset our patience counter
            best_value = current_value
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"[INFO]  Best model saved to: {best_model_path}")
            epochs_without_improvement = 0
        else:
            # Otherwise don't save the model, increment our patience counter and inform the user that the model hasn't improved
            epochs_without_improvement += 1
            tqdm.write(f"[INFO]  Early stopping: {epochs_without_improvement} / {cfg.patience} epochs without improvement of {cfg.monitor}")

        # Break out of the epoch loop if we hit our patience limit
        if epochs_without_improvement >= cfg.patience:
            tqdm.write(f"[INFO]  {cfg.monitor} has not improved in {cfg.patience} epochs - training stopped")
            break

        # Put the model back in training mode before the next iteration
        model.train()

if __name__ == "__main__":
    main()
