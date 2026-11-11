"""
Generates figures for the evaluation of models as created/evaluated by train.py/evaluate.py
"""

import json
import logging
import textwrap
from pathlib import Path

import hydra
import matplotlib
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")  # Draw to files only, don't bother with a window
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, PercentFormatter

log = logging.getLogger(__name__)

def plot_curves(history: list[dict], monitor: str, metric: str, title:str, out_path: Path) -> None:
    """
    Plots curves based on passed parameters
    :param history:     The history dictionary the curves will be plotted based on
    :param monitor:     The monitor value that determines the "best" epoch
    :param metric:      The metric being considered for the plot
    :param title:       The title to be used on the figure
    :param out_path:    The path the figure should be saved to
    :return:            None
    """

    # Determine best epoch
    if monitor == "valid_loss":
        best_row = min(history, key=lambda row: row[monitor])  # Lower validation loss is better
    else: 
        best_row = max(history, key=lambda row: row[monitor])  # Everything else is better if it's higher
    best_epoch = best_row["epoch"]

    # Get a list of all epochs and metrics being tracked
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    valid_losses = [row["valid_loss"] for row in history]
    metric_values = [row[metric] for row in history]

    # Create the figure
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(15, 6))

    # Draw lines
    ax_loss.plot(epochs, train_losses, marker="o", label="train")
    ax_loss.plot(epochs, valid_losses, marker="o", label="valid")
    ax_metric.plot(epochs, metric_values, marker="o", label=f"valid {metric}")

    # Shared panel stuff, best-epoch line, ticks, legend
    for ax in (ax_loss, ax_metric):
        ax.axvline(best_epoch, linestyle="--", color="grey", label=f"best epoch ({best_epoch}, {monitor})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

    # Non-shared panel stuff
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_metric.set(xlabel="epoch", ylabel=metric, title=f"Valid {metric}")

    # Add figure title
    fig.suptitle(title)

    # Save and close figs
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def short_count(n: int) -> str:
    """
    Shortens a count for display, e.g. 1186145 -> 1.19M
    :param n: The count to be shortened
    :return:  The shortened form of the count
    """

    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= size * 0.9995:  # Anything rounding up to 1000 of a given unit gets bumped up to the next unit
            return f"{n / size:.3g}{suffix}"  # If we're over a cutoff return the formatted count
    return str(n)  # If it's not above any of the cutoffs just return the string representation of the count

def split_shared(settings: list[str]) -> tuple[list[str], str]:
    """
    Splits setting names into parts; what's different between them and what do they all share
    :param settings:    Setting names as found in summary.json (kv pairs joined with +)
    :return:            The differing part of each name, and the shared parts as a single string
    """

    # Split the settings by individual settings
    parts = [setting.split("+") for setting in settings]

    # Determine what settings are shared among all
    shared = [part for part in parts[0] if all(part in other for other in parts[1:])]

    # Build shared + unique string
    short = ["+".join(part for part in setting_parts if part not in shared) or "all runs" for setting_parts in parts]
    return short, ", ".join(shared)

def stack_labels(labels: list[str], max_chars: int = 30) -> list[str]:
    """
    Puts each setting on a new line if any label is too long, so all labels match
    :param labels:       Labels made up of settings joined with "+"
    :param max_chars:   If any label is longer than this they all get stacked, defaults to 30
    :return:            The labels (either stacked or unchanged)
    """

    if any(len(label) > max_chars for label in labels):
        return [label.replace("+", "\n") for label in labels]
    return labels

def with_shared(title: str, shared: str) -> str:
    """
    Adds the settings shared by all bars to a title that gets wrapped so long lists don't widen figures
    :param title:   The figure title
    :param shared:  The shared settings (from split_shared())
    :return:        The title with the shared settings on the following line or lines
    """

    if not shared:
        return title
    return title + "\n" + textwrap.fill(f"shared: {shared}", width=80, break_on_hyphens=False)

def plot_comparison(rows: dict, metric: str, title: str, out_path: Path) -> None:
    """
    Plots comparison figures for multirun summaries for direct comparison between models with different settings 
    :param rows:        The rows in the summary table to use for generating comparison figures
    :param metric:      The metric being considered for comparison
    :param title:       The title to be used for the figure
    :param out_path:    The path the figure should be saved to
    :return:            None
    """

    # Determine names for bars in comparison and shared elements
    train_crops, settings = zip(*(name.split("/", 1) for name in rows))
    short, shared = split_shared(list(settings))
    if len(set(train_crops)) > 1:  # Only name training crop when the rows don't all share one
        short = [crop if setting == "all runs" else f"{crop}/{setting}" for crop, setting in zip(train_crops, short)]
    names = [f"{label}\n(n={row['runs']})" for label, row in zip(stack_labels(short), rows.values())]

    # Collect information from summary.json
    means = [row["stats"][metric]["mean"] for row in rows.values()]
    stds  = [row["stats"][metric]["std"]  for row in rows.values()]

    # Fixed bar area, each row gets room for its tallest label, and labels/title sit outside the axes
    lines = max(name.count("\n") + 1 for name in names)
    fig = plt.figure(figsize=(6, (0.25 * lines + 0.35) * len(names)))
    ax = fig.add_axes((0, 0, 1, 1))  # Axes fill the figure, savefig's bbox_inches="tight" grows the image to fit the labels

    # Use horizontal bar chart since we have long names that won't fit on a vertical bar chart very well
    bars = ax.barh(names, means, xerr=stds, capsize=4)
    ax.bar_label(bars, labels=[f"{m:.3f} ± {s:.3f}" for m, s in zip(means, stds)], label_type="center", color="white")
    ax.invert_yaxis()
    ax.set(xlabel=f"{metric} (mean ± std across seeds)", title=with_shared(title, shared))

    # Save and close fig
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_confusion(counts: dict, title: str, out_path: Path) -> None:
    """
    Plots confusion matrices based on the passed dictionary of counts
    :param counts:      The counts to be used in the confusion matrix
    :param title:       The title to be used for the figure
    :param out_path:    The path the figure should be saved to
    :return:            None
    """

    grid = [[counts["tp"], counts["fn"]],  # Actual sunlit:     predicted sunlit, predicted non-sunlit
            [counts["fp"], counts["tn"]]]  # Actual non-sunlit: predicted sunlit, predicted non-sunlit

    # Build cells
    shares = []
    labels = []

    # Calculate total
    total = sum(counts.values())

    # Guard against ZeroDivisionError (which should never actually happen but we might as well be safe)
    if total == 0:
        raise ValueError("No pixels counted")

    for row in grid:

        # Shares of whole split:
        row_shares = [value / total for value in row]
        shares.append(row_shares)

        # Labels for confusion matrix cells
        row_labels = [f"{short_count(value)}\n({value / total:.2%})" for value in row]
        labels.append(row_labels)

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw figure
    names = ["Sunlit Leaf", "Non-sunlit Leaf"]
    sns.heatmap(shares, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=1, cbar_kws={"format": PercentFormatter(xmax=1)}, annot_kws={"fontsize": 12}, square=True, xticklabels=names, yticklabels=names, ax=ax)

    # Format heatmap and set title
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set(xlabel="Predicted", ylabel="Actual")
    fig.suptitle(title)

    # Save and close fig
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_overview(results: dict, split: str, metric: str, out_path: Path) -> None:
    """
    Generates overview charts for the comparison of performance across crops
    :param results:     The dictionary holding summary results to use for chart creation
    :param split:       The split being compared
    :param metric:      The metric by which runs should be compared in the chart
    :param out_path:    Path the figure should be saved to
    :return:            None
    """

    # Create a table to store results
    table = {}  # table[setting][crop] = {"mean": ..., "std": ...}

    # Loop over results
    for section, rows in results.items():
        crop, section_split = section.rsplit("-", 1)  # Gives crop, section_split

        # Skip sections not matching `split`
        if section_split != split:
            continue

        # Get the crop the model was trained on and the settings used
        for name, row in rows.items():
            train_crop, setting = name.split("/", 1)

            # Skip cross-crop scores
            if train_crop != crop:
                continue

            # Store the necessary data 
            table.setdefault(setting, {})[crop] = row["stats"][metric]

    # Aggregate crops being evaluated and settings
    crops = sorted({crop for per_setting in table.values() for crop in per_setting})  # Abomination one liner, thanks Python
    settings = sorted(table)
    short, shared = split_shared(settings)
    labels = stack_labels(short)

    # If we only have one crop there's no point in running a comparison, so return early
    if len(crops) < 2:
        log.info(f"Could only locate metrics for one crop scored on {split}. Skipping overview.")
        return

    # Grouped bar figure info
    x = np.arange(len(crops))  # One slot per crop
    width = 0.8 / len(settings)  # Each group fills 80% of a slot so there's some gap
    fig = plt.figure(figsize=(max(6, 1.4 * len(crops), 0.25 * len(crops) * len(settings)), 5))  # Bar area gets wider with more crops and bars
    rotation = 90 if len(settings) > 2 else 0  # Value labels go vertical once bars get too thin for them
    ax = fig.add_axes((0, 0, 1, 1))  # Axes fill the figure, savefig's bbox_inches="tight" grows the image to fit the legend

    for i, setting in enumerate(settings):
        offset = (i - (len(settings) - 1) / 2) * width

        # Get per-crop stats
        stats = [table[setting].get(crop) for crop in crops]
        means = [stat["mean"] if stat else np.nan for stat in stats]
        stds = [stat["std"] if stat else 0 for stat in stats]

        # Build the bars
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=labels[i])
        ax.bar_label(bars, labels=[f"{mean:.3f}" if stat else "" for mean, stat in zip(means, stats)], label_type="center", color="white", fontsize=10, rotation=rotation)

    # Set info for graph
    ax.set_xticks(x, crops)
    ax.set(ylabel=f"{metric} (mean ± std across seeds)", title=with_shared(f"Scored on {split}: {metric} by crop", shared))
    ax.legend(title="setting", loc="upper left", bbox_to_anchor=(1.01, 1))  # Outside the plot so it can't cover bars
    ax.grid(axis="x", visible=False)

    # Save and close fig
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@hydra.main(config_path="conf", config_name="figures", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Generates figures based on model evaluation results
    :return: None
    """

    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    # Get the output directory
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {out_dir}")

    # Find all of the runs folders within the specified folder
    folder = Path(cfg.folder)
    runs = sorted([path.parent for path in folder.rglob("history.json")])
    log.info(f"{len(runs)} run(s) found")

    # If we don't find any runs we can just return early
    if not runs:
        log.warning(f"No runs found in {folder}")
        return

    for run in runs:
        run_name = run.relative_to(folder)  # Get relative path to run folder to use as a name
        metrics_files = sorted(run.glob("eval/*/metrics.json"))
        log.info(f"{run_name}: {len(metrics_files)} scored splits found")

        # Load the training history for the given run
        history_path = run / "history.json"
        with open(history_path, "r") as file:
            history = json.load(file)

        # Load model config for the given run
        config_path = run / ".hydra" / "config.yaml"
        settings = OmegaConf.load(config_path)

        # Raise a value error if we get an invalid metric
        if cfg.metric not in history[0]:
            raise ValueError(f"Metric {cfg.metric} not found in list of valid metrics: {sorted(history[0])}")

        # Determine the output folder and create it (with its parents too) if necessary
        run_out = out_dir / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        # Build title and generate figure
        title = f"{settings.crop.name}: {run.name}"
        plot_curves(history, settings.monitor, cfg.metric, title, run_out / "curves.png")

        # Plot confusion matrices for each metrics.json file
        for metrics_path in metrics_files:
            with open(metrics_path, "r") as file:
                data = json.load(file)

            # Extract values from data
            counts = data["metrics"]["counts"]
            evaluation = data["evaluation"]
            iou = data["metrics"]["scores"]["sunlit_iou"]

            # Append to title
            conf_title = title + f"\nscored on {evaluation['crop']} {evaluation['split']}, sunlit IoU {iou:.3f}"

            # Plot confusion matrices
            plot_confusion(counts, conf_title, run_out / f"confusion-{metrics_path.parent.name}.png")

    # Create comparison and overview charts - first we need to find the summary
    summary_path = folder / "summary" / "summary.json"
    if summary_path.is_file():
        with open(summary_path, "r") as file:
            summary = json.load(file)

        # Log a warning and skip making comparison charts for an invalid metric
        if cfg.metric not in summary["metrics"]:
            log.warning(f"{cfg.metric} not found in list of valid metrics: {summary['metrics']}; skipping comparison and overview charts")
        else:
            for section, rows in summary["results"].items():
                title = f"{section}: {cfg.metric} comparison"
                plot_comparison(rows, cfg.metric, title, out_dir / f"comparison-{section}.png")

            # Get list of splits
            splits = sorted({res.rsplit("-", 1)[1] for res in summary["results"]})
            for split in splits:
                plot_overview(summary["results"], split, cfg.metric, out_dir / f"overview-{split}.png")

    # Log a message letting the user know there was no summary found for the passed folder, we have nothing to generate charts from
    else:
        log.info(f"Comparison charts skipped, no summary.json found in {folder} - Use summarize.py to generate summaries")


if __name__ == "__main__":
    main()
