"""Shared utilities for filesystem operations, reporting, and image loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_open_image(path: Path) -> tuple[Image.Image | None, str | None]:
    """Open an image and return an error string instead of raising."""

    try:
        image = Image.open(path).convert("RGB")
        return image, None
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return None, str(exc)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe with a stable CSV index policy."""

    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk."""

    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def markdown_table_from_series(series: pd.Series, value_name: str) -> str:
    """Convert a series into a small markdown table."""

    lines = ["| key | value |", "| --- | ---: |"]
    for idx, value in series.items():
        lines.append(f"| {idx} | {value} |")
    return "\n".join(lines)


def summarize_dict(data: dict[str, Any]) -> str:
    """Render a dictionary as markdown bullet lines."""

    return "\n".join(f"- `{key}`: {value}" for key, value in data.items())


def finalize_report(output_dir: Path, config: Any, sections: list[dict[str, Any]]) -> None:
    """Build the final markdown report from section snippets."""

    lines = [
        f"# {config.report_title}",
        "",
        "## Run configuration",
        summarize_dict(
            {
                "dataset_root": config.dataset_root,
                "csv_path": config.csv_path,
                "output_dir": output_dir,
                "max_images": config.max_images,
            }
        ),
        "",
    ]

    for section in sections:
        title = section.get("title", "Untitled Section")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(section.get("markdown", "No summary available."))
        lines.append("")

    write_text(output_dir / "final_report.md", "\n".join(lines).strip() + "\n")
