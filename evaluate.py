"""
Script for the evaluation of trained segmentation models
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from components.scoring import load_checkpoint, score_run

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig):

    # Determine output directory
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # Load model and settings (it's already in eval mode from load_checkpoint)
    model, settings = load_checkpoint(Path(cfg.checkpoint))

    # We don't need to actually capture the return here
    score_run(model, settings, cfg.data_root, cfg.crop.name, cfg.split, out_dir)


if __name__ == "__main__":
    main()