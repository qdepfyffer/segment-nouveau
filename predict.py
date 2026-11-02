import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from components.dataset import IMAGE_SUFFIXES, _label_stem, classes, load_image
from components.inference import predict_logits
from components.progress import bar_desc
from components.scoring import load_checkpoint, run_folder

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("run_folder", lambda path: str(run_folder(Path(path))))

@hydra.main(config_path="conf", config_name="predict", version_base="1.3")
def main(cfg: DictConfig) -> None:

    # Get the output directory
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {out_dir}")

    # Load the model
    model, settings = load_checkpoint(Path(cfg.checkpoint), cfg.device)

    # Find input images to use for inference
    input_path = Path(cfg.input).expanduser()

    # If it's a directory we want all of the images inside of it
    if input_path.is_dir():
        imgs = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    # If it's a file and not a directory, treat it like a single image
    elif input_path.is_file():
        imgs = [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []

    # Otherwise just raise a FileNotFoundError
    else:
        raise FileNotFoundError(f"{input_path} not found")

    # Early return if we can't find any images to use for predictions
    if len(imgs) == 0:
        log.warning(f"{cfg.input}: no images found")
        return
    log.info(f"{cfg.input}: found {len(imgs)} images")

    device = next(model.parameters()).device
    palette = np.array([color for _, color in classes], dtype=np.uint8)

    # Determine paths for where mask output should be saved
    pred_masks_path = out_dir / "masks"
    pred_masks_path.mkdir(parents=True, exist_ok=True)

    # Determine paths for optional outputs and create directories if necessary
    pred_logits_path = out_dir / "logits"
    pred_overlays_path = out_dir / "overlays"
    if cfg.save_overlays:
        pred_overlays_path.mkdir(parents=True, exist_ok=True)
    if cfg.save_logits:
        pred_logits_path.mkdir(parents=True, exist_ok=True)

    # Prediction loop
    with torch.no_grad():
        for image_path in tqdm(imgs, desc=bar_desc(log, "[PREDICT] Progress")):

            # Load the image, unsqueeze() adds a batch dimension since that's what the model expects and move it to the specified device
            image = load_image(image_path).unsqueeze(0).to(device)

            # Get the logits, which can be optionally saved
            logits = predict_logits(model, image, settings.sizing)

            # The prediction is the argmax of the logits over the class dimension
            mask = logits.argmax(dim=1)[0].cpu().numpy()

            # Get the label stem for the image path (so predicted masks have names matching the gt labels)
            stem = _label_stem(image_path)

            # Optionally save logits
            if cfg.save_logits:
                np.save(pred_logits_path / f"{stem}_logits.npy", logits[0].cpu().numpy())

            # Swap each class ID in the mask for its color and save the predicted mask
            colored = palette[mask]
            Image.fromarray(colored).save(pred_masks_path / f"{stem}_pred.png")

            # Optionally save overlays (we need the colored prediction masks first)
            if cfg.save_overlays:
                original = Image.open(image_path).convert("RGB")
                overlay = Image.blend(original, Image.fromarray(colored), alpha=0.5)
                comp = Image.composite(overlay, original, Image.fromarray(mask == 1))
                comp.save(pred_overlays_path / f"{stem}_overlay.png")

    log.info(f"Wrote {len(imgs)} masks to {pred_masks_path} | Logits saved: {cfg.save_logits} ({pred_logits_path}) | Overlays saved: {cfg.save_overlays} ({pred_overlays_path})")

if __name__ == "__main__":
    main()
