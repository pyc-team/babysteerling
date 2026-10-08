import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from omegaconf import OmegaConf

CHECKPOINTS_DIR = "checkpoints"
OUTPUT_DIR = "reports/plots"

# Chart chrome, matching the house style (see dataviz skill: references/palette.md).
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Validated categorical palette, fixed order -> every model gets a stable color across all
# plots regardless of which subset FILTERS includes. Order follows MODEL_LABELS declaration
# order; models without a MODEL_LABELS entry get the remaining slots, alphabetically.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

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
        "grid.color": GRIDLINE,
        "font.family": "serif",
        "text.usetex": True,
    },
)

# Dotted path into each run's config.yaml -> allowed value(s). A run is included only if it
# matches every filter below. Leave empty to plot every run found under CHECKPOINTS_DIR.
FILTERS = {
    "model.wandb_name": {
        "dlm-steerling-bottleneck",
        "alm-concept-bottleneck",
        "alm-concept-residual",
        "alm-concept-naive-residual",
        "dlm-standard",
    },
}

# Raw model.wandb_name -> display name used for bar labels. Unmapped names are shown as-is.
MODEL_LABELS = {
    "dlm-standard": "DLM",
    "alm-concept-bottleneck": "ALM CBM",
    "dlm-steerling-bottleneck": "Steerling",
    "alm-concept-naive-residual": "ALM Naive Residual",
    "alm-concept-residual": "ALM Residual",
}

# Raw metrics.csv metric -> display name used for plot titles. Unmapped metrics are shown as-is.
METRIC_LABELS = {
    "val/macro_roc_auc": "Macro ROC-AUC",
    "val/lm_accuracy": "LM Accuracy",
    "val/macro_pr_auc": "Macro PR-AUC",
    "val/causal_concept_effect": "Causal Concept Effect",
    "val/total_loss": "Total Loss",
    "val/token_loss": "Token Loss",
    "val/concept_loss": "Concept Loss",
    "val/reconstruction_loss": "Reconstruction Loss",
    "val/independence_loss": "Independence Loss",
    "val/lm_residual_accuracy": "LM Accuracy (Residual)",
    "val/causal_concept_effect_concept": "Causal Concept Effect (Concept)",
    "val/token_residual_loss": "Token Loss (Residual)",
}

# Metric pairs rendered as ONE grouped barplot instead of two separate ones: each model gets two
# bars side by side (same color, second bar hatched) tucked close together, with a bigger gap
# between different models' pairs. Each tuple is (base_metric, variant_metric) -- raw metrics.csv
# names. A model missing one side of the pair just gets a single bar.
METRIC_GROUPS = [
    ("val/lm_accuracy", "val/lm_residual_accuracy"),
    ("val/causal_concept_effect", "val/causal_concept_effect_concept"),
    ("val/token_loss", "val/token_residual_loss"),
]
GROUPED_METRICS = {metric for pair in METRIC_GROUPS for metric in pair}

# Bar geometry, shared by every plot. BAR_WIDTH is the rendered width of EVERY single bar --
# solo or inside a pair -- so bars are always the same thickness no matter which chart they're
# in. PAIR_GAP is seaborn's own `gap` fraction between the two dodged bars inside one pair; to
# keep each of those bars at BAR_WIDTH despite the gap eating into them, the dodge call is given
# width=BAR_WIDTH * 2 / (1 - PAIR_GAP) (seaborn splits that across 2 hues, then shrinks by gap).
# Model ticks are 1 apart, so a pair's total footprint (2 * BAR_WIDTH + gap) must stay under 1 to
# avoid crowding neighboring models -- if you widen either constant, check for that.
BAR_WIDTH = 0.35
PAIR_GAP = 0.2
VARIANT_HATCH = "//"


_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """Escapes TeX special characters in labels that may come from raw config/metric names
    (MODEL_LABELS/METRIC_LABELS are opt-in, so unmapped names fall back to e.g. "val/token_loss"
    or "alm_residual", which would otherwise break usetex rendering)."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in text)


def discover_runs(checkpoints_dir: str) -> list[dict]:
    """Finds every checkpoint dir that has both metrics.csv and config.yaml."""
    runs = []
    pattern = os.path.join(checkpoints_dir, "*", "*", "metrics.csv")
    for metrics_path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(metrics_path)
        config_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(config_path):
            continue
        runs.append(
            {
                "config_hash": os.path.basename(run_dir),
                "cfg": OmegaConf.load(config_path),
                "metrics": pd.read_csv(metrics_path),
            }
        )
    return runs


def matches_filters(cfg, filters: dict) -> bool:
    for key, allowed in filters.items():
        allowed = {allowed} if isinstance(allowed, str) else set(allowed)
        if OmegaConf.select(cfg, key) not in allowed:
            return False
    return True


def label_run(cfg, config_hash: str, seen_labels: set) -> str:
    """Human-readable bar label, disambiguated with the hash if two runs share a name."""
    label = OmegaConf.select(cfg, "model.wandb_name") or config_hash
    label = MODEL_LABELS.get(label, label)
    if label in seen_labels:
        label = f"{label}-{config_hash[:6]}"
    seen_labels.add(label)
    return label


def build_dataframe(runs: list[dict]) -> pd.DataFrame:
    seen_labels = set()
    rows = []
    for run in runs:
        label = label_run(run["cfg"], run["config_hash"], seen_labels)
        for _, row in run["metrics"].iterrows():
            rows.append({"run": label, "metric": row["metric"], "value": row["value"]})
    return pd.DataFrame(rows)


def build_model_colors(run_labels: set) -> dict:
    """Assigns each model a fixed categorical color, stable across plots/filters."""
    ordered_names = list(MODEL_LABELS.values())
    remaining = sorted(run_labels - set(ordered_names))
    ordered = [name for name in ordered_names if name in run_labels] + remaining
    return {
        name: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        for i, name in enumerate(ordered)
    }


def style_axes(ax, output_label: str) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=1, color=GRIDLINE)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlabel("")
    ax.set_ylabel(latex_escape(output_label), color=PRIMARY_INK)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_text(latex_escape(label.get_text()))


def plot_combined_metric_group(
    df: pd.DataFrame, metrics: tuple, output_dir: str, model_colors: dict
) -> None:
    """One grouped barplot for a (base_metric, variant_metric) pair: each model gets both
    bars tucked together (same color, variant hatched) if it has both metrics, else one bar.
    """
    base_metric, variant_metric = metrics
    sub = df[df["metric"].isin(metrics)]
    if sub.empty:
        return

    ordered_models = [m for m in model_colors if m in sub["run"].unique()]
    base_models = set(sub.loc[sub["metric"] == base_metric, "run"])
    variant_models = set(sub.loc[sub["metric"] == variant_metric, "run"])
    paired_models = base_models & variant_models
    has_variant = bool(variant_models)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Paired models: let seaborn's own dodge+gap place the two bars (no manual x math).
    if paired_models:
        sns.barplot(
            data=sub[sub["run"].isin(paired_models)],
            x="run",
            y="value",
            order=ordered_models,
            hue="metric",
            hue_order=list(metrics),
            dodge=True,
            gap=PAIR_GAP,
            width=BAR_WIDTH * 2 / (1 - PAIR_GAP),
            legend=False,
            ax=ax,
        )
        # Only remaining job: color by model (not by hue) and hatch the variant container.
        for metric, container in zip(metrics, ax.containers):
            is_variant = metric == variant_metric
            models_in_container = [m for m in ordered_models if m in paired_models]
            for patch, model in zip(container.patches, models_in_container):
                patch.set_facecolor(model_colors[model])
                patch.set_hatch(VARIANT_HATCH if is_variant else None)
                patch.set_edgecolor(PRIMARY_INK if is_variant else model_colors[model])
                patch.set_linewidth(0.6 if is_variant else 0)
            ax.bar_label(container, fmt="%.3f", color=SECONDARY_INK, padding=3)

    # Solo models (only the base metric): plain centered bars, same width as any other chart.
    # A model that somehow had only the variant metric would be dropped here -- doesn't happen
    # in practice, since every METRIC_GROUPS variant is reported alongside its base metric.
    solo_models = [m for m in ordered_models if m not in paired_models]
    if solo_models:
        solo_sub = sub[sub["run"].isin(solo_models) & (sub["metric"] == base_metric)]
        sns.barplot(
            data=solo_sub,
            x="run",
            y="value",
            order=ordered_models,
            dodge=False,
            width=BAR_WIDTH,
            legend=False,
            ax=ax,
        )
        container = ax.containers[-1]
        for patch, model in zip(container.patches, solo_models):
            patch.set_facecolor(model_colors[model])
            patch.set_edgecolor(model_colors[model])
        ax.bar_label(container, fmt="%.3f", color=SECONDARY_INK, padding=3)

    style_axes(ax, METRIC_LABELS.get(base_metric, base_metric))

    if has_variant:
        legend_handles = [
            Patch(
                facecolor=MUTED_INK,
                edgecolor=MUTED_INK,
                label=latex_escape(METRIC_LABELS.get(base_metric, base_metric)),
            ),
            Patch(
                facecolor=MUTED_INK,
                edgecolor=PRIMARY_INK,
                hatch=VARIANT_HATCH,
                linewidth=0.6,
                label=latex_escape(METRIC_LABELS.get(variant_metric, variant_metric)),
            ),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="best", fontsize=9)

    fig.tight_layout()
    suffix = "_combined.png" if has_variant else ".png"
    fname = base_metric.replace("/", "_") + suffix
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_colors = build_model_colors(set(df["run"].unique()))

    for pair in METRIC_GROUPS:
        plot_combined_metric_group(df, pair, output_dir, model_colors)

    for metric, group in df.groupby("metric"):
        if metric in GROUPED_METRICS:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=group,
            x="run",
            y="value",
            order=[m for m in model_colors if m in group["run"].unique()],
            hue="run",
            palette=model_colors,
            legend=False,
            width=BAR_WIDTH,
            ax=ax,
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", color=SECONDARY_INK, padding=3)
        style_axes(ax, METRIC_LABELS.get(metric, metric))

        fig.tight_layout()
        fname = metric.replace("/", "_") + ".png"
        fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    runs = discover_runs(CHECKPOINTS_DIR)
    runs = [r for r in runs if matches_filters(r["cfg"], FILTERS)]
    if not runs:
        print("No runs found under", CHECKPOINTS_DIR, "matching FILTERS.")
        return

    df = build_dataframe(runs)
    plot_metrics(df, OUTPUT_DIR)
    print(
        f"Wrote {df['metric'].nunique()} plots to {OUTPUT_DIR}/ for {len(runs)} runs."
    )


if __name__ == "__main__":
    main()
