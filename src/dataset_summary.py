"""Dataset summary tables, plots, and markdown reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from config import AnalysisConfig
from src.utils import markdown_table_from_series, save_dataframe, write_text


def _plot_class_counts(class_counts: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    class_counts.sort_values(ascending=False).plot(kind="bar", color="#2d6a4f")
    plt.title("Class Counts")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_size_distributions(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df["width"].dropna(), bins=30, color="#577590")
    axes[0].set_title("Image Width")
    axes[1].hist(df["height"].dropna(), bins=30, color="#43aa8b")
    axes[1].set_title("Image Height")
    axes[2].hist(df["aspect_ratio"].dropna(), bins=30, color="#f3722c")
    axes[2].set_title("Aspect Ratio")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_dataset_summary(
    df: pd.DataFrame, ingest_context: dict[str, Any], config: AnalysisConfig
) -> dict[str, Any]:
    """Create summary artefacts and a markdown snippet."""

    readable_df = df[df["is_readable"]].copy()
    class_counts = (
        readable_df["label"].value_counts().sort_values(ascending=False).rename_axis("label")
    )
    duplicates = (
        readable_df["filename"]
        .value_counts()
        .rename_axis("filename")
        .reset_index(name="count")
    )
    duplicates = duplicates[duplicates["count"] > 1]

    size_summary = readable_df[["width", "height", "aspect_ratio", "file_size_bytes"]].describe()
    save_dataframe(class_counts.reset_index(name="count"), config.output_dir / "class_counts.csv")
    save_dataframe(size_summary.reset_index(), config.output_dir / "image_size_summary.csv")
    save_dataframe(duplicates, config.output_dir / "duplicate_filenames.csv")

    _plot_class_counts(class_counts, config.output_dir / "class_counts.png")
    _plot_size_distributions(readable_df, config.output_dir / "image_size_distributions.png")

    summary_lines = [
        f"- Number of images: `{len(df)}`",
        f"- Readable images: `{len(readable_df)}`",
        f"- Number of classes: `{readable_df['label'].nunique()}`",
        f"- Duplicate filenames: `{len(duplicates)}`",
        f"- Missing/corrupted images: `{int((~df['is_readable']).sum())}`",
        "",
        "### Class counts",
        markdown_table_from_series(class_counts, "count"),
        "",
        "### Ingestion note",
        ingest_context["markdown"],
    ]

    write_text(config.output_dir / "dataset_summary.md", "\n".join(summary_lines) + "\n")
    return {
        "title": "Dataset Summary",
        "markdown": "\n".join(summary_lines),
    }
