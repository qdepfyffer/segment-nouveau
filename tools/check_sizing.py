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

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from components.dataset import SegmentationDataset
from components.metrics import scores_from_counts
from components.scoring import count_split, load_checkpoint
from models.padding import pad_to_multiple


def pad_to_64(images):
    return pad_to_multiple(images, 64)  # Internal resizing of models commonly uses some power of 2, we use 64 because it's evenly divisible by 32, 16, 8, 4, and 2, so the model should handle it well


def pad_black(images):
    return torch.nn.functional.pad(images, (0, 32, 0, 32))  # Check whether the model loses the plot when it sees black padding


@hydra.main(config_path="../conf", config_name="check_sizing", version_base="1.3")
def main(cfg: DictConfig):

    log = logging.getLogger(__name__)

    # Load the model
    model, settings = load_checkpoint(Path(cfg.checkpoint), cfg.device)

    # There's no point in checking sizing if the model was trained using a stretch transform since it tests padding at the native image size
    if settings.sizing.name != "native":  # If you add some other non-stretch transform you might need to edit this to allow check_sizing to work with it
        log.warning(f"check_sizing.py only tests at native sizing, and this model was trained with {settings.sizing.name}")
        return

    # Load and build scoring dataset (batch size 1 lets us score every image at native resolution)
    dataset_root = Path(cfg.data_root) if cfg.data_root is not None else Path(settings.data_root)
    eval_ds = SegmentationDataset(dataset_root, cfg.crop.name, cfg.split)  # Split comes from conf/check_sizing.yaml (valid by default)
    eval_loader = DataLoader(eval_ds, batch_size=1)

    # Dictionary for pad variants
    variants = {
        "as is": None,
        "mirrored to multiple of 64": pad_to_64,
        "black padding of 32 px": pad_black,
    }

    # Run evals
    for name, prepare in variants.items():
        totals = count_split(model, eval_loader, settings.sizing, prepare)
        scores = scores_from_counts(**totals)
        log.info(f"{name:30s} sunlit IoU: {scores['sunlit_iou']:.4f}")

if __name__ == "__main__":
    main()