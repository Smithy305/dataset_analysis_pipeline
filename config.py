"""Configuration objects for the dataset analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass(slots=True)
class AnalysisConfig:
    """Runtime configuration for loading a dataset and writing outputs."""

    dataset_root: Optional[Path] = None
    csv_path: Optional[Path] = None
    path_root: Optional[Path] = None
    output_dir: Path = Path("outputs/default_run")
    image_column: str = "image_path"
    label_column: str = "label"
    source_column: Optional[str] = None
    max_images: Optional[int] = None
    recursive: bool = True
    valid_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )
    image_size_for_embedding: int = 224
    embedding_batch_size: int = 32
    random_seed: int = 42
    montage_image_size: int = 180
    montage_columns: int = 5
    top_k_extremes: int = 20
    outlier_top_k: int = 25
    duplicate_top_k: int = 25
    duplicate_hamming_threshold: int = 6
    quality_metric_columns: tuple[str, ...] = (
        "brightness",
        "contrast",
        "blur_score",
        "saturation",
        "white_pixel_ratio",
        "bright_region_count",
        "entropy",
    )
    embedding_prefixes: tuple[str, ...] = ("embedding_",)
    report_title: str = "Prototype Dataset Analysis Report"
    extra_metadata_columns: Iterable[str] = field(default_factory=tuple)

    def validate(self) -> None:
        """Validate the minimum configuration needed to run the pipeline."""

        if not self.dataset_root and not self.csv_path:
            raise ValueError("Provide either dataset_root or csv_path.")
        if self.dataset_root:
            self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        if self.csv_path:
            self.csv_path = Path(self.csv_path).expanduser().resolve()
        if self.path_root:
            self.path_root = Path(self.path_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
