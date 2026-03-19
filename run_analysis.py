"""Command-line entrypoint for the dataset analysis pipeline."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dataset_analysis_pipeline_mpl"))

from config import AnalysisConfig
from src.bias_analysis import run_bias_analysis
from src.data_loader import load_dataset_dataframe
from src.dataset_summary import generate_dataset_summary
from src.duplicate_detection import run_duplicate_detection
from src.embeddings import run_embedding_analysis
from src.image_quality import compute_image_quality_metrics
from src.multiview_placeholder import write_multiview_placeholder
from src.outlier_detection import run_outlier_detection
from src.utils import ensure_dir, finalize_report, save_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prototype image dataset analysis.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Root folder for folder-per-class dataset.")
    parser.add_argument("--csv-path", type=Path, default=None, help="CSV manifest containing image paths and labels.")
    parser.add_argument("--path-root", type=Path, default=None, help="Optional prefix for relative paths in CSV mode.")
    parser.add_argument("--image-column", type=str, default="image_path", help="Image path column for CSV mode.")
    parser.add_argument("--label-column", type=str, default="label", help="Label column for CSV mode.")
    parser.add_argument("--source-column", type=str, default=None, help="Optional source/domain column.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default_run"), help="Output directory.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for fast prototype runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        dataset_root=args.dataset_root,
        csv_path=args.csv_path,
        path_root=args.path_root,
        image_column=args.image_column,
        label_column=args.label_column,
        source_column=args.source_column,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )
    config.validate()

    ensure_dir(config.output_dir)

    df, ingest_context = load_dataset_dataframe(config)
    save_dataframe(df, config.output_dir / "dataset_manifest.csv")

    summary_context = generate_dataset_summary(df, ingest_context, config)
    df, quality_context = compute_image_quality_metrics(df, config)
    save_dataframe(df, config.output_dir / "dataset_manifest_with_quality.csv")

    bias_context = run_bias_analysis(df, config)
    df, duplicate_context = run_duplicate_detection(df, config)
    df, embedding_context = run_embedding_analysis(df, config)
    save_dataframe(df, config.output_dir / "dataset_manifest_with_embeddings.csv")

    df, outlier_context = run_outlier_detection(df, config)
    save_dataframe(df, config.output_dir / "dataset_manifest_final.csv")

    multiview_context = write_multiview_placeholder(config.output_dir)
    finalize_report(
        output_dir=config.output_dir,
        config=config,
        sections=[
            summary_context,
            quality_context,
            bias_context,
            duplicate_context,
            embedding_context,
            outlier_context,
            multiview_context,
        ],
    )


if __name__ == "__main__":
    main()
