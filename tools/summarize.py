"""
Summarizes metrics for a given directory containing one or more metrics.json file(s) in itself or subdirectories.
Writes summary/summary.json and summary/summarize.log within the given directory
"""

import json
import logging
import statistics
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="summarize", version_base="1.3")
def main(cfg: DictConfig):

    # Find all `metrics.json` files for a given sweep
    metrics_paths = sorted(Path(cfg.folder).rglob("metrics.json"))
    count = len(metrics_paths)
    log.info(f"Found {count} total metrics.json files in {cfg.folder}.")
    if count == 0:
        log.warning(f"No metrics.json files found under {cfg.folder}. Please make sure you've provided a folder containing at least one metrics.json file.")
        return

    # Get the output directory
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {out_dir}")

    # Empty dict for storing run groups
    groups = {}

    # Loop over the metrics paths
    for path in metrics_paths:
        with open(path) as file:
            data = json.load(file)

        # Get information for summarization
        eval_crop = data["evaluation"]["crop"]  # The crop the model was scored on
        eval_split = data["evaluation"]["split"]  # The split the model was scored on
        train_crop = data["model"]["crop"]["name"]  # The crop the model was trained on
        scores = data["metrics"]["scores"]  # The model's scores {"accuracy": 0.000..., "precision": ..., ...}
        run_folder = path.parents[2]  # The run folder (metrics.json sits in <run>/eval/<crop>-<split>/)

        # Strip the "seed-n" portion of the run folder (since we're averaging across seeds for comparison purposes)
        pieces = [piece for piece in run_folder.name.split("+") if not piece.startswith("seed-")]
        stripped = "+".join(pieces)
        setting = stripped if stripped else "default"  # If a run ONLY differs across seeds "stripped" would be empty

        # Group the runs
        key = (eval_crop, eval_split, train_crop, setting)
        group = groups.setdefault(key, {"scores":[], "runs": []})
        group["scores"].append(scores)
        group["runs"].append(run_folder)

    for key, group in groups.items():

        # Calculate mean and stdev for each score (this will be for each setting across seeds)
        group["stats"] = {}
        for name in cfg.metrics:

            # Validate metric names (do they appear in the first run's scores?)
            if name not in group["scores"][0]:
                raise ValueError(f"{name} not found in list of valid metrics: {sorted(group['scores'][0])}")

            # Aggregate score values and calculate mean and stdev
            values = [scores[name] for scores in group["scores"]]
            mean_scores = statistics.mean(values)
            std_scores = 0.0 if len(values) < 2 else statistics.stdev(values)

            # Store the mean and stdev in the group
            group["stats"][name] = {"mean": mean_scores, "std": std_scores}

    # Work out the width for the name column and set the padding value for other columns
    label = "trained on / setting"
    name_width = max([len(f"{key[2]}/{key[3]}") for key in groups] + [len(label)])
    PADDING = 17  # Constant value to use for column padding

    # Dict to store summary info for writing summary.json
    summary = {"folder": str(cfg.folder), "metrics": list(cfg.metrics), "results": {}}

    # Build the summary table and dict
    current_section = None # So we can print a header if the section ever changes
    for key in sorted(groups):

        # Unpack the values and set our new section
        eval_crop, eval_split, train_crop, setting = key
        group = groups[key]
        section = (eval_crop, eval_split)

        # Print the table header each time our new section differs from the current section
        if section != current_section:
            log.info(f"Scored on {eval_crop} {eval_split}")
            log.info(f"{label:<{name_width}} {'runs':>6}" + "".join([f"{metric:>{PADDING}}" for metric in cfg.metrics]))
            current_section = section

        row_name = f"{train_crop}/{setting}"  # The name for the row (first column, trained on / setting)
        runs = len(group["runs"])  # Number of runs with the given config
        run_folders = [str(run.relative_to(cfg.folder)) for run in group["runs"]]  # Relative paths to run folders
        stats = group["stats"]  # The stats dictionary for each group
        metrics_cells = []  # Empty list to store the cells for each metric as they're built

        # Build the cells, pad each one, and append it to the list
        for name in cfg.metrics:
            mean = group["stats"][name]["mean"]
            std = group["stats"][name]["std"]
            cell = f"{mean:.4f} ± {std:.4f}"
            metrics_cells.append(f"{cell:>{PADDING}}")

        # Log the resulting row, name, number of runs with that configuration, and all the metrics
        log.info(f"{row_name:<{name_width}} {runs:>6}" + "".join(metrics_cells))

        # Add to summary dict
        section_name = f"{eval_crop}-{eval_split}"
        section_results = summary["results"].setdefault(section_name, {})
        section_results[row_name] = {"runs": runs, "run_folders": run_folders, "stats": stats}

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=4)
    log.info(f"Summary written to {summary_path}.")


if __name__ == "__main__":
    main()