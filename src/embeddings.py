"""Visual embedding extraction and 2D projection plots."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from config import AnalysisConfig
from src.utils import save_dataframe, safe_open_image, write_text
from src.visualizations import plot_embedding_scatter

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency branch
    umap = None


class ImagePathDataset(Dataset):
    """Dataset wrapper that loads images from paths listed in a dataframe."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=False)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image, _ = safe_open_image(Path(row["image_path"]))
        if image is None:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(image), int(row["index"])


def _build_embedding_model() -> tuple[torch.nn.Module, str]:
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        status = "pretrained"
    except Exception:
        model = models.resnet18(weights=None)
        status = "random_init"
    model.fc = torch.nn.Identity()
    model.eval()
    return model, status


def _extract_embeddings(
    df: pd.DataFrame, config: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    readable_df = df[df["is_readable"]].copy()
    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size_for_embedding, config.image_size_for_embedding)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImagePathDataset(readable_df, transform)
    loader = DataLoader(dataset, batch_size=config.embedding_batch_size, shuffle=False, num_workers=0)

    model, model_status = _build_embedding_model()
    features: list[np.ndarray] = []
    indices: list[int] = []
    with torch.no_grad():
        for batch, batch_indices in tqdm(loader, desc="Extracting embeddings"):
            output = model(batch).cpu().numpy()
            features.append(output)
            indices.extend(batch_indices.tolist())

    embedding_matrix = np.concatenate(features, axis=0) if features else np.empty((0, 512))
    ordered = np.array(indices, dtype=int)
    return ordered, embedding_matrix, model_status


def _project_embeddings(embedding_matrix: np.ndarray, random_seed: int) -> dict[str, np.ndarray]:
    projections: dict[str, np.ndarray] = {}
    if len(embedding_matrix) == 0:
        return projections

    projections["pca"] = PCA(n_components=2, random_state=random_seed).fit_transform(embedding_matrix)
    if umap is not None and len(embedding_matrix) >= 5:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"n_jobs value 1 overridden to 1 by setting random_state.*",
                category=UserWarning,
            )
            projections["umap"] = umap.UMAP(
                n_components=2,
                random_state=random_seed,
                n_jobs=1,
            ).fit_transform(embedding_matrix)
    elif len(embedding_matrix) >= 5:
        perplexity = max(2, min(30, len(embedding_matrix) // 4))
        projections["tsne"] = TSNE(
            n_components=2,
            random_state=random_seed,
            init="pca",
            perplexity=perplexity,
        ).fit_transform(embedding_matrix)
    return projections


def run_embedding_analysis(df: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract embeddings, project them to 2D, and write plots."""

    ordered_indices, embedding_matrix, model_status = _extract_embeddings(df, config)
    if len(embedding_matrix) == 0:
        return df, {
            "title": "Embedding Analysis",
            "markdown": "No readable images were available for embedding extraction.",
        }

    embedding_columns = [f"embedding_{dim:03d}" for dim in range(embedding_matrix.shape[1])]
    embedding_frame = pd.DataFrame(np.nan, index=df.index, columns=embedding_columns)
    embedding_frame.loc[ordered_indices, embedding_columns] = embedding_matrix
    df = pd.concat([df, embedding_frame], axis=1)

    projections = _project_embeddings(embedding_matrix, config.random_seed)
    embedding_dir = config.output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    for projection_name, coords in projections.items():
        x_col = f"{projection_name}_x"
        y_col = f"{projection_name}_y"
        projection_frame = pd.DataFrame(np.nan, index=df.index, columns=[x_col, y_col])
        projection_frame.loc[ordered_indices, [x_col, y_col]] = coords
        df = pd.concat([df, projection_frame], axis=1)

        for color_col in ("label", "brightness", "blur_score", "analysis_domain", "source_domain"):
            if color_col in df.columns and df[color_col].notna().any():
                plot_embedding_scatter(
                    df,
                    x_col=x_col,
                    y_col=y_col,
                    color_col=color_col,
                    output_path=embedding_dir / f"{projection_name}_by_{color_col}.png",
                    title=f"{projection_name.upper()} coloured by {color_col}",
                )

    save_columns = ["image_path", "label", "source_domain", "analysis_domain", "analysis_domain_kind"] + [
        column
        for column in df.columns
        if column.startswith("embedding_") or column.endswith("_x") or column.endswith("_y")
    ]
    save_dataframe(df[save_columns], embedding_dir / "embedding_table.csv")

    projection_names = ", ".join(projections.keys()) if projections else "none"
    markdown = (
        f"- Embedding model status: `{model_status}`\n"
        f"- Readable images embedded: `{len(embedding_matrix)}`\n"
        f"- Projection methods available: `{projection_names}`\n"
        f"- Embedding plots written to `{embedding_dir}`"
    )
    write_text(embedding_dir / "embedding_summary.md", markdown + "\n")
    return df, {
        "title": "Embedding Analysis",
        "markdown": markdown,
    }
