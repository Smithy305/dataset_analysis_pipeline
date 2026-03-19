"""Dataset ingestion for folder-based and CSV-based image datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from config import AnalysisConfig
from src.utils import safe_open_image, save_dataframe


def _infer_folder_label(dataset_root: Path, image_path: Path) -> tuple[str, str | None]:
    relative_parts = image_path.relative_to(dataset_root).parts
    if len(relative_parts) >= 2:
        return relative_parts[-2], relative_parts[0] if len(relative_parts) > 2 else None
    return "unlabelled", None


def _build_records_from_folder(config: AnalysisConfig) -> list[dict[str, Any]]:
    pattern = "**/*" if config.recursive else "*"
    image_paths = [
        path
        for path in sorted(config.dataset_root.glob(pattern))
        if path.is_file() and path.suffix.lower() in config.valid_extensions
    ]
    if config.max_images:
        image_paths = image_paths[: config.max_images]

    records: list[dict[str, Any]] = []
    for path in tqdm(image_paths, desc="Scanning images"):
        label, source_domain = _infer_folder_label(config.dataset_root, path)
        records.append(
            {
                "image_path": str(path.resolve()),
                "label": label,
                "source_domain": source_domain,
            }
        )
    return records


def _resolve_csv_image_path(raw_value: str, path_root: Path | None) -> Path:
    path = Path(str(raw_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path_root:
        return (path_root / path).resolve()
    return path.resolve()


def _build_records_from_csv(config: AnalysisConfig) -> list[dict[str, Any]]:
    frame = pd.read_csv(config.csv_path)
    if config.image_column not in frame.columns or config.label_column not in frame.columns:
        raise ValueError(
            f"CSV must contain `{config.image_column}` and `{config.label_column}` columns."
        )

    if config.max_images:
        frame = frame.head(config.max_images).copy()

    records: list[dict[str, Any]] = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Reading CSV rows"):
        record = row.to_dict()
        record["image_path"] = str(_resolve_csv_image_path(record[config.image_column], config.path_root))
        record["label"] = record[config.label_column]
        if config.source_column and config.source_column in row:
            record["source_domain"] = row[config.source_column]
        records.append(record)
    return records


def _enrich_records_with_file_metadata(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Inspecting file metadata"):
        path = Path(record["image_path"])
        image, error = safe_open_image(path)
        if image is None:
            errors.append(
                {
                    "image_path": str(path),
                    "label": record.get("label", "unknown"),
                    "error": error,
                }
            )
            record = {
                **record,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "file_size_bytes": path.stat().st_size if path.exists() else None,
                "is_readable": False,
                "load_error": error,
            }
        else:
            width, height = image.size
            record = {
                **record,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4) if height else None,
                "file_size_bytes": path.stat().st_size,
                "is_readable": True,
                "load_error": None,
            }
        enriched.append(record)

    return pd.DataFrame(enriched), pd.DataFrame(errors)


def _build_analysis_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Infer a lightweight analysis domain when explicit metadata is absent."""

    df = df.copy()
    df["resolution_bucket"] = df.apply(
        lambda row: f"{int(row['width'])}x{int(row['height'])}"
        if pd.notna(row["width"]) and pd.notna(row["height"])
        else "unknown",
        axis=1,
    )

    has_explicit_domain = "source_domain" in df.columns and df["source_domain"].notna().any()
    if has_explicit_domain:
        df["analysis_domain"] = df["source_domain"].fillna("unknown")
        df["analysis_domain_kind"] = "explicit_source_domain"
    else:
        df["analysis_domain"] = df["resolution_bucket"]
        df["analysis_domain_kind"] = "inferred_resolution_bucket"
    return df


def load_dataset_dataframe(config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load image metadata and return the working dataframe for the pipeline."""

    if config.csv_path:
        records = _build_records_from_csv(config)
        ingest_mode = "csv"
    else:
        records = _build_records_from_folder(config)
        ingest_mode = "folder"

    df, error_df = _enrich_records_with_file_metadata(records)
    if "source_domain" not in df.columns:
        df["source_domain"] = None
    df = _build_analysis_domain(df)

    df["filename"] = df["image_path"].map(lambda value: Path(value).name)
    save_dataframe(error_df, config.output_dir / "corrupted_or_unreadable_images.csv")

    context = {
        "title": "Data Ingestion",
        "markdown": (
            f"Ingestion mode: `{ingest_mode}`.\n\n"
            f"- Total rows discovered: `{len(df)}`\n"
            f"- Readable images: `{int(df['is_readable'].sum())}`\n"
            f"- Unreadable images: `{int((~df['is_readable']).sum())}`\n"
            f"- Corrupted image log: `corrupted_or_unreadable_images.csv`\n"
            f"- Analysis domain kind: `{df['analysis_domain_kind'].iloc[0]}`"
        ),
    }
    return df, context
