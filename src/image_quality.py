"""Image-quality feature extraction and related artefacts."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import AnalysisConfig
from src.utils import save_dataframe, safe_open_image, write_text
from src.visualizations import create_image_montage, plot_metric_by_class, plot_metric_histograms


def _compute_entropy(gray_image: np.ndarray) -> float:
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / np.maximum(histogram.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _compute_quality_features(image: Image.Image) -> dict[str, float]:
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(hsv[:, :, 1].mean())

    white_mask = np.all(rgb >= 245, axis=2).astype(np.uint8)
    white_pixel_ratio = float(white_mask.mean())
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    bright_region_count = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 20)) if num_labels > 1 else 0
    entropy = _compute_entropy(gray)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "saturation": saturation,
        "white_pixel_ratio": white_pixel_ratio,
        "bright_region_count": bright_region_count,
        "entropy": entropy,
    }


def _save_extreme_metric_tables(df: pd.DataFrame, metric: str, output_dir: Path, top_k: int) -> None:
    metric_df = df[["image_path", "label", metric]].dropna().sort_values(metric)
    save_dataframe(metric_df.head(top_k), output_dir / f"{metric}_lowest.csv")
    save_dataframe(metric_df.tail(top_k).sort_values(metric, ascending=False), output_dir / f"{metric}_highest.csv")


def compute_image_quality_metrics(
    df: pd.DataFrame, config: AnalysisConfig
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add per-image quality metrics and generate summary artefacts."""

    readable_mask = df["is_readable"].fillna(False)
    for metric in config.quality_metric_columns:
        if metric not in df.columns:
            df[metric] = np.nan

    for idx, row in tqdm(df[readable_mask].iterrows(), total=int(readable_mask.sum()), desc="Computing quality metrics"):
        image, error = safe_open_image(Path(row["image_path"]))
        if image is None:
            df.loc[idx, "load_error"] = error
            df.loc[idx, "is_readable"] = False
            continue
        metrics = _compute_quality_features(image)
        for metric_name, metric_value in metrics.items():
            df.loc[idx, metric_name] = metric_value

    quality_dir = config.output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_histograms(df, config.quality_metric_columns, quality_dir)

    for metric in config.quality_metric_columns:
        plot_metric_by_class(df, metric, quality_dir / f"{metric}_by_class.png")
        _save_extreme_metric_tables(df, metric, quality_dir, config.top_k_extremes)

        lowest = df[["image_path", metric]].dropna().sort_values(metric).head(config.top_k_extremes)
        highest = df[["image_path", metric]].dropna().sort_values(metric, ascending=False).head(config.top_k_extremes)
        create_image_montage(
            lowest["image_path"].tolist(),
            quality_dir / f"{metric}_lowest_montage.png",
            titles=[f"{metric}={value:.2f}" for value in lowest[metric].tolist()],
            image_size=config.montage_image_size,
            ncols=config.montage_columns,
        )
        create_image_montage(
            highest["image_path"].tolist(),
            quality_dir / f"{metric}_highest_montage.png",
            titles=[f"{metric}={value:.2f}" for value in highest[metric].tolist()],
            image_size=config.montage_image_size,
            ncols=config.montage_columns,
        )

    metric_summary = df[list(config.quality_metric_columns)].describe().round(3)
    save_dataframe(metric_summary.reset_index(), quality_dir / "quality_metric_summary.csv")

    correlation = df[list(config.quality_metric_columns)].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation.index)), correlation.index)
    plt.colorbar(label="Correlation")
    plt.title("Quality Metric Correlation")
    plt.tight_layout()
    plt.savefig(quality_dir / "quality_metric_correlation.png", dpi=180)
    plt.close()

    summary_text = (
        "Computed per-image quality metrics for readable images.\n\n"
        f"- Metrics: {', '.join(config.quality_metric_columns)}\n"
        f"- Histograms written to `{quality_dir}`\n"
        f"- Extremes tables and montages written to `{quality_dir}`"
    )
    write_text(quality_dir / "quality_summary.md", summary_text + "\n")
    return df, {
        "title": "Image Quality Analysis",
        "markdown": summary_text,
    }
