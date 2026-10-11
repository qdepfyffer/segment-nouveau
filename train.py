"""
Script for the training of segmentation models
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset
from components.losses import build_loss
from models import build_model, MODELS
from components.optimizers import build_optimizer, OPTIMIZERS
from components.transforms import training_transform, PadToSize


def parse_args():
    """
    Contains all the command line arguments used in training with reasonable defaults for any that aren't explicitly required.
    :return: Command line arguments to be used in model training
    """
    parser = argparse.ArgumentParser(
        description="Train a semantic segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_root", type=Path, required=True, help="Root dataset directory, should have a subdirectory for the crop you want to train on")
    parser.add_argument("--crop", type=str, required=True, help="The crop (name of subdirectory within the dataset root directory) you want to train on")
    parser.add_argument("--architecture", type=str, choices=list(MODELS), default="deeplabv3", help="Which model to train")
    parser.add_argument("--optimizer", type=str, choices=list(OPTIMIZERS), default="rmsprop", help="Which optimizer to use during training")
    parser.add_argument("--dice_weight", type=float, default=0.0, help="Determines dice weight to be used for loss - cross-entropy plus dice weight multiplied by dice loss - 0 to disable dice loss")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs to train for (passes over the training set)")
    parser.add_argument("--patience", type=int, default=7, help="How many epochs to continue training for without seeing improvement in validation loss")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images in each training / evaluation batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate of the model")
    parser.add_argument("--train_crop", nargs=2, type=int, default=[256, 480], help="The size of the random crop that is applied to training images (h x w)")
    parser.add_argument("--eval_pad", nargs=2, type=int, default=[288, 512], help="Pads validation images to the given size (h x w)")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"), help="The folder this script should output to")
    parser.add_argument("--seed", type=int, default=42, help="Number to use as a random seed for anything that can be seeded")
    return parser.parse_args()

def main() -> None:
    """
    This is where the training loop lives.
    :return: None
    """
    # Get command line args
    args = parse_args()

    # Use our RNG seed
    torch.manual_seed(args.seed)

    # Determine output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.output_dir / args.crop / timestamp
    out_dir.mkdir(parents=True)
    best_model_path = out_dir / f"{args.architecture}-best.pt"

    # Save settings (passed args) to aid in reproducibility
    settings = vars(args)
    with open(out_dir / "settings.json", "w") as file:
        json.dump(settings, file, indent=4, default=str)

    # Build specified model
    model = build_model(args.architecture, num_classes=2)
    model = model.to("cuda")  # Move model to GPU

    # Load our datasets
    train_ds = SegmentationDataset(args.data_root, args.crop, "train",
                                   transform=training_transform(args.train_crop[0], args.train_crop[1]))
    valid_ds = SegmentationDataset(args.data_root, args.crop, "valid",
                                   transform=PadToSize(args.eval_pad[0], args.eval_pad[1]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)

    # Set optimizer and loss function
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    loss_fn = build_loss(args.dice_weight)

    # Begin model training
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    model.train()
    for epoch in range(args.epochs):
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

        tqdm.write(f"[TRAIN] Loss: {(total_train_loss / len(train_loader)):.4f}")

        # Validation loop
        total_valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"[VALID] Epoch {epoch + 1}"):

                # Put images and masks on GPU
                images = images.to("cuda")
                masks = masks.to("cuda")

                logits = model(images)
                loss = loss_fn(logits, masks)
                total_valid_loss += loss.item()

        tqdm.write(f"[VALID] Loss: {(total_valid_loss / len(valid_loader)):.4f}")

        # Check if we've gotten a better validation loss than our current best
        if total_valid_loss < best_valid_loss:
            # If we have a better validation loss update our current best, save the current weights, inform the user, and reset our patience counter
            best_valid_loss = total_valid_loss
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"[INFO]  Best model saved to: {best_model_path}")
            epochs_without_improvement = 0
        else:
            # Otherwise don't save the model, increment our patience counter and inform the user that the model hasn't improved
            epochs_without_improvement += 1
            tqdm.write(f"[INFO]  Early stopping: {epochs_without_improvement} / {args.patience} epochs without improvement")

        # Break out of the epoch loop if we hit our patience limit
        if epochs_without_improvement >= args.patience:
            tqdm.write(f"[INFO]  Validation loss has not improved in {args.patience} epochs - training stopped.")
            break

        # Put the model back in training mode before the next iteration
        model.train()

if __name__ == "__main__":
    main()
