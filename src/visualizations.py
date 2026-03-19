"""Plotting and montage helpers."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import ensure_dir, safe_open_image


def plot_metric_histograms(df: pd.DataFrame, metrics: Sequence[str], output_dir: Path) -> None:
    """Save one histogram per metric."""

    ensure_dir(output_dir)
    for metric in metrics:
        values = df[metric].dropna()
        if values.empty:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=30, color="#3a86ff", alpha=0.85)
        plt.title(f"{metric} distribution")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_hist.png", dpi=180)
        plt.close()


def plot_metric_by_class(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a class-wise boxplot for a metric."""

    plot_df = df[["label", metric]].dropna()
    if plot_df.empty:
        return
    labels = sorted(plot_df["label"].unique())
    data = [plot_df.loc[plot_df["label"] == label, metric].values for label in labels]
    plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric} by class")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def create_image_montage(
    image_paths: Sequence[str],
    output_path: Path,
    titles: Sequence[str] | None = None,
    image_size: int = 180,
    ncols: int = 5,
) -> None:
    """Create a simple grid montage of local images."""

    valid_images: list[Image.Image] = []
    valid_titles: list[str] = []
    for idx, image_path in enumerate(image_paths):
        image, error = safe_open_image(Path(image_path))
        if image is None:
            continue
        valid_images.append(image.resize((image_size, image_size)))
        if titles:
            valid_titles.append(titles[idx])
        else:
            valid_titles.append(Path(image_path).name)

    if not valid_images:
        return

    nrows = ceil(len(valid_images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes_array = np.array(axes).reshape(-1)
    for axis in axes_array:
        axis.axis("off")

    for axis, image, title in zip(axes_array, valid_images, valid_titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=8)
        axis.axis("off")

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_embedding_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    output_path: Path,
    title: str,
) -> None:
    """Create a 2D scatter plot for embedding coordinates."""

    plot_df = df[[x_col, y_col, color_col]].dropna()
    if plot_df.empty:
        return

    plt.figure(figsize=(7, 6))
    if pd.api.types.is_numeric_dtype(plot_df[color_col]):
        scatter = plt.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=plot_df[color_col],
            cmap="viridis",
            s=10,
            alpha=0.8,
        )
        plt.colorbar(scatter, label=color_col)
    else:
        categories = plot_df[color_col].astype(str)
        unique_values = categories.unique()
        cmap = plt.cm.get_cmap("tab20", len(unique_values))
        for idx, value in enumerate(unique_values):
            subset = plot_df[categories == value]
            plt.scatter(subset[x_col], subset[y_col], s=10, alpha=0.8, label=value, color=cmap(idx))
        plt.legend(markerscale=2, fontsize=8, loc="best")

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
