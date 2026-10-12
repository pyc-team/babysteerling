import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import OmegaConf

from plot_metrics import (
    BASELINE,
    CHECKPOINTS_DIR,
    MODEL_LABELS,
    MUTED_INK,
    OUTPUT_DIR,
    PRIMARY_INK,
    SURFACE,
    latex_escape,
    savefig,
)

# Each concept_topk_heatmap.json has one column per concept (~100+) -- too many to fit in a
# paper figure. Every plot keeps only the TOP_N_CONCEPTS concepts with the strongest #1 (top-1)
# weight, i.e. the ones with the single clearest attributed token.
TOP_N_CONCEPTS = 20

# How many top-k ranks (rows) to show, out of however many the JSON has (usually 10) -- ranks are
# already ordered strongest-first, so this just keeps the first TOP_N_TOKENS of them.
TOP_N_TOKENS = 5

# Magnitude encoding (cell color ~ weight) -> one hue, light to dark, not a rainbow. Reuses the
# house categorical blue as the single hue so it reads as part of the same palette family.
SEQUENTIAL_HUE = "#2a78d6"

sns.set_theme(
    style="ticks",
    rc={
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": MUTED_INK,
        "text.color": PRIMARY_INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "font.family": "serif",
        "text.usetex": True,
    },
)


def discover_heatmaps(checkpoints_dir: str) -> list[dict]:
    """Finds every run's concept_topk_heatmap.json under checkpoints_dir (only runs whose
    bottleneck overrides LM.diagnostics() write one -- e.g. residual/naive_residual). Also reads
    the run's own config.yaml, saved alongside it, for model.wandb_name (see label_model below).
    """
    runs = []
    pattern = os.path.join(checkpoints_dir, "*", "concept_topk_heatmap.json")
    for heatmap_path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(heatmap_path)
        config_hash = os.path.basename(run_dir)
        with open(heatmap_path) as f:
            data = json.load(f)
        config_path = os.path.join(run_dir, "config.yaml")
        wandb_name = None
        if os.path.exists(config_path):
            wandb_name = OmegaConf.select(
                OmegaConf.load(config_path), "model.wandb_name"
            )
        runs.append({"config_hash": config_hash, "wandb_name": wandb_name, **data})
    return runs


def label_model(run: dict) -> str:
    """Same model.wandb_name -> display name mapping as plot_metrics.label_run, falling back to
    the raw wandb_name (or the config hash if that's missing too) when unmapped."""
    return MODEL_LABELS.get(run["wandb_name"], run["wandb_name"] or run["config_hash"])


def _slug(text: str) -> str:
    """Filesystem-safe version of a label, for use in a filename."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def select_top_concepts(run: dict, top_n_concepts: int, top_n_tokens: int) -> dict:
    """Keeps the top_n_concepts concepts with the strongest #1 (top-1) weight, descending, and
    the top_n_tokens strongest ranks, then transposes to rank-rows x concept-columns (the
    orientation this plot renders)."""
    weights = np.array(run["weights"])[
        :, :top_n_tokens
    ]  # shape: [n_concepts, top_n_tokens]
    order = np.argsort(-weights[:, 0])[:top_n_concepts]
    tokens = run["tokens"]
    ranks = run["columns"][:top_n_tokens]
    return {
        "concepts": [run["concepts"][i] for i in order],
        "ranks": ranks,
        "weights": weights[order].T,  # shape: [top_n_tokens, top_n_concepts]
        "tokens": [
            [latex_escape(tokens[i][r]) for i in order] for r in range(len(ranks))
        ],
    }


def plot_heatmap(
    run: dict, top_n_concepts: int, top_n_tokens: int, output_dir: str
) -> str:
    selected = select_top_concepts(run, top_n_concepts, top_n_tokens)
    weights = np.asarray(selected["weights"])
    concept_labels = [latex_escape(c) for c in selected["concepts"]]
    rank_labels = [latex_escape(r) for r in selected["ranks"]]

    fig_width = max(6.0, 0.6 * len(concept_labels))
    fig_height = 0.5 * len(rank_labels) + 1.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        weights,
        ax=ax,
        cmap=sns.light_palette(SEQUENTIAL_HUE, as_cmap=True),
        annot=np.array(selected["tokens"], dtype=object),
        fmt="",
        annot_kws={"fontsize": 7, "color": PRIMARY_INK},
        xticklabels=concept_labels,
        yticklabels=rank_labels,
        linewidths=2,
        linecolor=SURFACE,
        cbar_kws={"label": "attribution weight"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    label = label_model(run)
    ax.set_title(latex_escape(label))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"concept_topk_heatmap_{_slug(label)}")
    fig.tight_layout()
    savefig(fig, out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    runs = discover_heatmaps(CHECKPOINTS_DIR)
    if not runs:
        print("No concept_topk_heatmap.json found under", CHECKPOINTS_DIR)
        return
    for run in runs:
        out_path = plot_heatmap(run, TOP_N_CONCEPTS, TOP_N_TOKENS, OUTPUT_DIR)
        print(f"Wrote {out_path}.png and {out_path}.pdf")


if __name__ == "__main__":
    main()
