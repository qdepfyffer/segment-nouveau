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
    log.info(f"Log for this evaluation: {out_dir}")

    # Find all checkpoint paths
    checkpoint_paths = sorted(Path(cfg.folder).rglob("*-best.pt"))
    runs = [path.parent for path in checkpoint_paths]

    # Exit early if we can't find any valid runs
    if len(runs) == 0:
        log.warning(f"No run folders found within {cfg.folder}. Please make sure you've provided a folder containing at least one valid run.")
        return

    # Log the number of found runs
    log.info(f"Runs: {len(runs)}")

    # List to keep track of failed run scorings
    failed = []

    # Score all the found runs
    for number, run in enumerate(runs, start=1):
        log.info(f"[{number}/{len(runs)}] Scoring {run}")

        # Determine where the run's scores go
        run_out = run / "eval" / f"{cfg.crop.name}-{cfg.split}"

        handler = None
        if run_out.resolve() != out_dir.resolve():  # A single run already logs here through Hydra
            run_out.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(run_out / "evaluate.log", mode="w")
            handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
            logging.getLogger().addHandler(handler)
        try:
            # Load the run that is currently being scored
            model, settings = load_checkpoint(run)

            # Score the run
            score_run(model, settings, cfg.data_root, cfg.crop.name, cfg.split, run_out)

            # Free memory before we start the next run otherwise bigger models are going to hurt
            del model, settings

        except Exception:
            log.exception(f"Scoring failed for {run} - continuing from the next run...")
            failed.append(run)

        finally:
            if handler is not None:
                logging.getLogger().removeHandler(handler)
                handler.close()

    # Log diagnostic information
    log.info(f"Scored {len(runs) - len(failed)} of {len(runs)} runs.")
    if len(failed) > 0:
        for failure in failed:
            log.warning(f"Scoring failed: {failure}")


if __name__ == "__main__":
    main()