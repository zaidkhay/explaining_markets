"""Correlation diagnostics and plots for joined live-event predictions.

Diagnostic only (CLAUDE.md §7.1, §13): correlation here is not the scored
objective (Delta R2, see CLAUDE.md §2) and nothing in this module fits,
selects, or promotes anything. It exists to visualize how predicted and
realized percentiles for live events line up, and how that agreement trends
as more live events accumulate -- not evidence that the model itself is
improving, since coefficients are never refit on live events.

Input is the ``recent_joined.csv`` produced by
``scripts/evaluate_recent_from_archive.py`` (or any CSV/DataFrame with the
same ``predicted_percentile`` / ``realized_percentile`` columns).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("predicted_percentile", "realized_percentile")


def load_joined_rows(path: str | Path) -> pd.DataFrame:
    """Load a joined predictions/outcomes CSV, dropping incomplete rows.

    Sorts chronologically by ``event_datetime`` when that column is present,
    since :func:`expanding_correlation` assumes row order is chronological.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df["predicted_percentile"] = df["predicted_percentile"].astype(float)
    df["realized_percentile"] = df["realized_percentile"].astype(float)

    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce", utc=True)
        df = df.sort_values("event_datetime", kind="stable", na_position="last")

    return df.reset_index(drop=True)


def correlation_summary(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Pearson and Spearman correlation between predicted and realized percentiles."""
    n = len(df)
    if n < 2:
        return {"n": n, "pearson": None, "spearman": None}
    pearson = df["predicted_percentile"].corr(df["realized_percentile"], method="pearson")
    spearman = df["predicted_percentile"].corr(df["realized_percentile"], method="spearman")
    return {
        "n": n,
        "pearson": None if pd.isna(pearson) else float(pearson),
        "spearman": None if pd.isna(spearman) else float(spearman),
    }


def expanding_correlation(df: pd.DataFrame, min_periods: int = 3) -> pd.Series:
    """Chronological expanding-window Pearson correlation.

    Each point uses every row up to and including it, in ``df`` row order
    (call :func:`load_joined_rows` first so that order is chronological).
    NaN until ``min_periods`` rows are available.
    """
    return df["predicted_percentile"].expanding(min_periods=min_periods).corr(
        df["realized_percentile"]
    )


def plot_correlation(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Live events: prediction vs realized",
    min_periods: int = 3,
) -> Path:
    """Render a scatter (predicted vs realized) plus an expanding-correlation trend panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = correlation_summary(df)
    trend = expanding_correlation(df, min_periods=min_periods)

    fig, (scatter_ax, trend_ax) = plt.subplots(1, 2, figsize=(11, 4.5))

    scatter_ax.scatter(df["predicted_percentile"], df["realized_percentile"], alpha=0.7, edgecolor="none")
    scatter_ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="y = x")
    if summary["n"] >= 2:
        slope, intercept = np.polyfit(df["predicted_percentile"], df["realized_percentile"], 1)
        xs = np.linspace(0, 1, 50)
        scatter_ax.plot(xs, slope * xs + intercept, color="C1", linewidth=1.5, label="best fit")
    scatter_ax.set_xlim(0, 1)
    scatter_ax.set_ylim(0, 1)
    scatter_ax.set_xlabel("predicted percentile")
    scatter_ax.set_ylabel("realized percentile")
    r_txt = "n/a" if summary["pearson"] is None else f"{summary['pearson']:.3f}"
    rho_txt = "n/a" if summary["spearman"] is None else f"{summary['spearman']:.3f}"
    scatter_ax.set_title(f"n={summary['n']}  pearson r={r_txt}  spearman rho={rho_txt}")
    scatter_ax.legend(loc="upper left", fontsize=8)

    trend_ax.plot(np.arange(1, len(trend) + 1), trend.to_numpy(), marker="o", markersize=3)
    trend_ax.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    trend_ax.set_xlabel("live events, chronological")
    trend_ax.set_ylabel("expanding pearson correlation")
    trend_ax.set_title("Correlation trend as live events accumulate")
    trend_ax.set_ylim(-1.05, 1.05)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
