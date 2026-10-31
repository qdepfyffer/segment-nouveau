"""
Generates figures for the evaluation of models as created/evaluated by train.py/evaluate.py
"""

import json
import logging
from pathlib import Path

import hydra
import matplotlib
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

def plot_comparison(rows: dict, metric: str, title: str, out_path: Path) -> None:
    """
    Plots comparison figures for multirun summaries for direct comparison between models with different settings 
    :param rows:        The rows in the summary table to use for generating comparison figures
    :param metric:      The metric being considered for comparison
    :param title:       The title to be used for the figure
    :param out_path:    The path the figure should be saved to
    :return:            None
    """

    # Collect information from summary.json
    names = [f"{name}\n(n={row['runs']})" for name, row in rows.items()]
    means = [row["stats"][metric]["mean"] for row in rows.values()]
    stds  = [row["stats"][metric]["std"]  for row in rows.values()]

    # Create the figure - more runs needs a bigger figure, adjust the below constant values if things don't look quite right
    fig, ax = plt.subplots(figsize=(10, 0.9 * len(names) + 1.5))

    # Use horizontal bar chart since we have long names that won't fit on a vertical bar chart very well
    bars = ax.barh(names, means, xerr=stds, capsize=4)
    ax.bar_label(bars, labels=[f"{m:.3f} ± {s:.3f}" for m, s in zip(means, stds)], label_type="center", color="white")
    ax.invert_yaxis()
    ax.set(xlabel=f"{metric} (mean ± std across seeds)", title=title)

    # Save and close fig
    fig.tight_layout()
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

    # Create comparison bar chart - first we need to find the summary
    summary_path = folder / "summary" / "summary.json"
    if summary_path.is_file():
        with open(summary_path, "r") as file:
            summary = json.load(file)

        # Log a warning and skip making comparison charts for an invalid metric
        if cfg.metric not in summary["metrics"]:
            log.warning(f"{cfg.metric} not found in list of valid metrics: {summary['metrics']}; skipping comparison chart")
        else:
            for section, rows in summary["results"].items():
                title = f"{section}: {cfg.metric} comparison"
                plot_comparison(rows, cfg.metric, title, out_dir / f"comparison-{section}.png")

    # Log a message letting the user know there was no summary found for the passed folder, we have nothing to generate charts from
    else:
        log.info(f"Comparison charts skipped, no summary.json found in {folder} - Use summarize.py to generate summaries")


if __name__ == "__main__":
    main()
