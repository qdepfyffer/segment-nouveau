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
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.dataset import SegmentationDataset
from components.git_info import git_info
from components.inference import predict_logits
from components.losses import build_loss
from components.metrics import confusion_counts, scores_from_counts
from components.optimizers import build_optimizer
from components.progress import bar_desc
from components.scoring import load_checkpoint, score_run
from components.transforms import build_training_transform
from models import build_model

# Set up logging
log = logging.getLogger(__name__)

# Options that don't change how a model is being trained, so they're ok to vary when a run is resumed
RUN_ONLY_KEYS = {"resume", "data_root", "output_dir", "device", "num_workers", "eval_splits"}


# Pulled out logic for safe file saving since we use it in two places now
def save_safely(contents, path: Path) -> None:
    """
    Saves to a temp file first and then swaps it in so nothing can ever read a half written file
    :param contents:    Whatever should be saved via torch.save
    :param path:        Where the file should end up
    :return:            None
    """

    temp_path = path.with_suffix(".tmp")  # Temp file, lives next to the real one - this only creates a path, no file exists yet
    torch.save(contents, temp_path)
    temp_path.replace(path)  # .replace() instead of .rename() - .rename() fails on Windows if the file already exists


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    This is where the training loop lives.
    :return: None
    """

    # Make sure a valid monitor argument has been passed
    MONITORS = ["valid_loss", "sunlit_iou", "f1", "mean_iou"]
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

    # Determine output directory and path to most recent set of weights
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    best_model_path = out_dir / f"{cfg.model.name}-best.pt"
    last_path = out_dir / f"{cfg.model.name}-last.pt"  # Everything needed to carry on training if the run gets interrupted

    # Capture current git repo commit hash and whether or not the repo is dirty (uncommitted changes)
    revision = git_info()
    log.info(f"Current revision: {revision['commit']}, uncommitted changes: {revision['dirty']}")
    if revision["dirty"]:
        # Log this warning so it's explicitly clear this run (likely) won't be reproducible with the code corresponding to the current commit hash
        log.warning("Current repo has uncommitted changes. The commit hash alone does not describe this run's code.")

    # Store a .json file holding the current revision information
    with open(out_dir / "git.json", "w") as file:
        json.dump(revision, file, indent=4)

    # Log which GPU is being used for the current run so that we can see if we're using the wrong GPU or a busy GPU in the first few lines of the log
    device = torch.device(cfg.device)
    if device.type == "cuda":

        # Get the properties for the current device if its type is CUDA
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)

        # Get free memory and total memory
        free, total = torch.cuda.mem_get_info(index)

        # Get bus ID as shown by nvidia-smi and log a message listing the free and total memory
        bus_id = f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0"
        log.info(f"Using cuda:{index}: {props.name}, bus ID {bus_id}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")

        # Inform the user if less than half of the GPU's memory is available (means it's being used by something else when the script starts running)
        if free < total / 2:
            log.warning("Less than half of this GPU's memory is currently available. Another process may be using the same GPU. Check nvidia-smi for more information.")

    # Otherwise if we're not using a CUDA device just log the device name, doesn't make sense to run the above checks otherwise
    else:
        log.info(f"Using device: {device}")

    # Build specified model
    model = build_model(cfg.model, num_classes=2)
    model = model.to(cfg.device)  # Move model to specified device

    # Load our datasets
    train_ds = SegmentationDataset(Path(cfg.data_root), cfg.crop.name, "train",
                                   transform=build_training_transform(cfg.sizing, cfg.crop.train_crop))
    valid_ds = SegmentationDataset(Path(cfg.data_root), cfg.crop.name, "valid")
    train_loader = DataLoader(train_ds, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, num_workers=cfg.num_workers, batch_size=1)  # Validate with a batch size of 1 so images don't need to be the same size

    # Set optimizer and loss function
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    loss_fn = build_loss(cfg.loss)

    # Initialize best_value to negative infinity, we use a higher-is-better approach for monitoring model performance
    # Loss will need to be negated during the training for this calculation
    best_value = float("-inf")

    # Keep track of best epoch number
    best_epoch = None

    # Initialize history as an empty list so we can track the model's scores and plot training curves later
    history = []

    # Begin model training
    epochs_without_improvement = 0
    start_epoch = 0  # Which epoch the training loop starts from - only changes when runs are resumed

    # Settings that need to match for a run to be resumed
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)  # A DictConfig always converts to a dict so this just shuts up the type checker
    training_settings = {key: value for key, value in config.items() if key not in RUN_ONLY_KEYS}

    # Pick up where interrupted run left off
    if cfg.resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=True)  # Load the most recently completed epoch into state

        # Refuse to resume if any settings that impact training results have changed since the run started
        different = sorted(key for key in state["settings"].keys() | training_settings.keys() if state["settings"].get(key) != training_settings.get(key))
        if different:
            raise ValueError(f"Can't resume {out_dir} - these settings differ from when it started: {', '.join(different)}")
        if state["code"] != revision:
            log.warning(f"This run was started on commit {state['code']['commit']} (uncommitted changes: {state['code']['dirty']}) - the rest of it will be trained on the current revision")

        # Put everything back the way it was at the end of the most recently completed epoch
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])  # Same shuffling and random crops as an uninterrupted run would get
        best_value = state["best_value"]
        best_epoch = state["best_epoch"]
        epochs_without_improvement = state["epochs_without_improvement"]
        history = state["history"]

        # A run that already stopped early has nothing left to train
        start_epoch = state["epoch"] if epochs_without_improvement < cfg.patience else cfg.epochs
        log.info(f"Resuming after epoch {state['epoch']} (best epoch so far: {best_epoch})")
    elif cfg.resume:
        log.warning(f"resume=true but there's no {last_path.name} in {out_dir} - starting from scratch")

    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        # Training loop
        total_train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=bar_desc(log, f"[TRAIN] Epoch {epoch + 1}")):

            # Put images and masks on specified device
            images = images.to(cfg.device)
            masks = masks.to(cfg.device)

            logits = model(images)  # We don't need to use predict_logits logic here because images have already been transformed, so predict_logits would essentially be a no-op here
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
            for images, masks in tqdm(valid_loader, desc=bar_desc(log, f"[VALID] Epoch {epoch + 1}")):

                # Put images and masks on specified device
                images = images.to(cfg.device)
                masks = masks.to(cfg.device)

                # Get model's predictions and calculate loss
                logits = predict_logits(model, images, cfg.sizing)
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

        # The formatting was bugging me so this will make things look a little better hopefully
        PAD_VALUE = 15

        log.info(f"[VALID] {'Loss:':<{PAD_VALUE}} {valid_loss:.4f}\t{'Mean IoU:':<{PAD_VALUE}} {scores['mean_iou']:.4f}")
        log.info(f"[VALID] {'Sunlit IoU:':<{PAD_VALUE}} {scores['sunlit_iou']:.4f}\t{'Non-Sunlit IoU:':<{PAD_VALUE}} {scores['non_sunlit_iou']:.4f}")
        log.info(f"[VALID] {'Precision:':<{PAD_VALUE}} {scores['precision']:.4f}\t{'Accuracy:':<{PAD_VALUE}} {scores['accuracy']:.4f}")
        log.info(f"[VALID] {'Recall:':<{PAD_VALUE}} {scores['recall']:.4f}\t{'F1:':<{PAD_VALUE}} {scores['f1']:.4f}")

        # Check if we've gotten a better score than our current best for the monitored value
        if current_value > best_value:
            # If we have a better score, update our current best, save the current weights, inform the user, and reset our patience counter
            best_value = current_value

            # Update best epoch
            best_epoch = epoch + 1

            # Save the weights safely so that nothing can ever read a half written set of weights
            save_safely(model.state_dict(), best_model_path)

            log.info(f"Best model saved to: {best_model_path}")
            epochs_without_improvement = 0
        else:
            # Otherwise don't save the model, increment our patience counter and inform the user that the model hasn't improved
            epochs_without_improvement += 1
            log.info(f"Early stopping: {epochs_without_improvement} / {cfg.patience} epochs without improvement of {cfg.monitor}")

        # Save everything necessary to carry on training from this point if it is interrupted
        save_safely({
            "settings": training_settings,
            "code": revision,
            "epoch": epoch + 1,  # Number of finished epochs
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_value": best_value,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "rng": torch.get_rng_state(),
        }, last_path)

        # Break out of the epoch loop if we hit our patience limit
        if epochs_without_improvement >= cfg.patience:
            log.info(f"{cfg.monitor} has not improved in {cfg.patience} epochs - training stopped")
            break

        # Put the model back in training mode before the next iteration
        model.train()

    # Say which epoch's weights were kept, since the last "Best model saved" line can be far up the log
    if best_epoch is None:
        log.warning(f"{cfg.monitor} never produced a usable value - no weights saved")
    else:
        log.info(f"Trained for {len(history)} epoch(s) - best epoch {best_epoch} ({cfg.monitor}: {history[best_epoch - 1][cfg.monitor]:.4f})")

    del model, optimizer  # Drop the training copy & everything the optimizer holds so it doesn't sit in memory alongside the loaded best checkpoint for evaluation
    torch.cuda.empty_cache()  # Hand back freed memory

    if not eval_splits:
        return  # Exit here if we're not running auto evaluation of the final trained model (no eval_split specified by conf/train.yaml or user override in CLI)

    # Otherwise we try to load the best model and evaluate it over it's OWN dataset (evaluate.py can still evaluate a model on different datasets)
    try:
        if not best_model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")  # Exit early if the checkpoint doesn't exist for any reason - Caught below, logged, then re-raised
        model, settings = load_checkpoint(out_dir, cfg.device)
        for split in eval_splits:
            split_dir = out_dir / "eval" / f"{cfg.crop.name}-{split}"
            if cfg.resume and (split_dir / "metrics.json").is_file():
                log.info(f"Already scored on {split} - skipping")
                continue  # A resumed run that was already scored should keep its scores
            score_run(model, settings, None, cfg.crop.name, split, split_dir, num_workers=cfg.num_workers)
    except Exception:
        log.exception("Automatic scoring failed - training results intact, score them with evaluate.py")
        raise  # Mark this run as failed, sweep should still finish its other runs and report failures at the end (re-throws the error that was just caught)

if __name__ == "__main__":
    main()
