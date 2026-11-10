"""
Prints a model's parameter count, checks output size for oddly sized inputs, and measures peak GPU memory per training step.
Run this after writing a new model and before training it.
"""

import logging

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from components.optimizers import build_optimizer
from models import build_model

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="model_info", version_base="1.3")
def main(cfg: DictConfig):

    # Log model config before we try to build so we can see what config was responsible even if building fails
    log.info(f"Model config: {OmegaConf.to_yaml(cfg.model)}")

    # Build model
    model = build_model(cfg.model, num_classes=2).to(cfg.device)

    # Calculate total parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Log info
    log.info(f"Total parameters: {total:,}; Trainable parameters: {trainable:,}")

    # Check shape for oddly sized inputs (using pistachio's native size), output must come back at the same size
    model.eval()
    with torch.no_grad():

        # Get output of the model for one random rgb image
        image = torch.rand(1, 3, 342, 504, device=cfg.device)
        logits = model(image)

    # Note the input and output sizes, raise a ValueError if they don't match
    log.info(f"Input {tuple(image.shape)} -> output {tuple(logits.shape)}")
    if logits.shape[-2:] != image.shape[-2:]:
        raise ValueError(f"Output size {tuple(logits.shape[-2:])} does not match input size {tuple(image.shape[-2:])} - check size_multiple and the crop-back in the forward function of the model")

    # Only check memory if we're running on the GPU
    if not cfg.device.startswith("cuda"):
        log.info("Not running on GPU, memory check skipped.")
        return

    # Create a fake training batch at the configured size with random images and 0/1 labels
    height, width = cfg.size
    model.train()
    optimizer = build_optimizer("rmsprop", model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats(cfg.device)
    images = torch.rand(cfg.batch_size, 3, height, width, device=cfg.device)
    masks = torch.randint(0, 2, (cfg.batch_size, height, width), device=cfg.device)

    # One full training step (like in train.py)
    try:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), masks)
        loss.backward()
        optimizer.step()
    except torch.cuda.OutOfMemoryError:
        log.warning(f"Out of memory at batch {cfg.batch_size}, {height}x{width} - try a smaller batch_size")
        return

    # Report peak against what the GPU actually has, leave some headroom, shoot for ~85% utilization 
    peak = torch.cuda.max_memory_allocated(cfg.device) / 1024**3
    available = torch.cuda.get_device_properties(cfg.device).total_memory / 1024**3
    log.info(f"Peak memory for one training step at batch {cfg.batch_size}, {height}x{width}: {peak:.2f} of {available:.2f} GiB")


if __name__ == "__main__":
    main()
