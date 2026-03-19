"""Prototype outlier detection for unusual or suspicious dataset samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import AnalysisConfig
from src.utils import save_dataframe, write_text
from src.visualizations import create_image_montage


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    min_value = np.min(values)
    max_value = np.max(values)
    if np.isclose(max_value, min_value):
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("embedding_") and column[10:].isdigit()]


def run_outlier_detection(df: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run several simple outlier detectors and produce a ranked suspicious-sample table."""

    quality_columns = [column for column in config.quality_metric_columns if column in df.columns]
    embedding_columns = _embedding_columns(df)
    candidate_df = df.dropna(subset=quality_columns).copy()
    if candidate_df.empty:
        return df, {
            "title": "Outlier Detection",
            "markdown": "No quality features were available, so outlier analysis was skipped.",
        }

    scaler = StandardScaler()
    quality_matrix = scaler.fit_transform(candidate_df[quality_columns])

    iso = IsolationForest(random_state=config.random_seed, contamination="auto")
    isolation_score = -iso.fit(quality_matrix).score_samples(quality_matrix)
    candidate_df["outlier_quality_isolation"] = _normalize_scores(isolation_score)

    if embedding_columns:
        embedding_matrix = candidate_df[embedding_columns].to_numpy()
        nn = NearestNeighbors(n_neighbors=min(6, len(candidate_df)))
        nn.fit(embedding_matrix)
        distances, _ = nn.kneighbors(embedding_matrix)
        candidate_df["outlier_embedding_nn"] = _normalize_scores(distances[:, -1])

        centroid_scores = []
        for _, row in candidate_df.iterrows():
            label_subset = candidate_df[candidate_df["label"] == row["label"]]
            centroid = label_subset[embedding_columns].mean(axis=0).to_numpy()
            point = row[embedding_columns].to_numpy(dtype=float)
            centroid_scores.append(float(np.linalg.norm(point - centroid)))
        candidate_df["outlier_class_centroid"] = _normalize_scores(np.asarray(centroid_scores))
    else:
        candidate_df["outlier_embedding_nn"] = 0.0
        candidate_df["outlier_class_centroid"] = 0.0

    candidate_df["outlier_score"] = candidate_df[
        ["outlier_quality_isolation", "outlier_embedding_nn", "outlier_class_centroid"]
    ].mean(axis=1)

    reasons: list[str] = []
    for _, row in candidate_df.iterrows():
        flags = []
        if row["outlier_quality_isolation"] >= 0.8:
            flags.append("quality features are unusual")
        if row["outlier_embedding_nn"] >= 0.8:
            flags.append("embedding is far from nearest neighbours")
        if row["outlier_class_centroid"] >= 0.8:
            flags.append("embedding is far from its class centroid")
        reasons.append("; ".join(flags) if flags else "mildly unusual across detectors")
    candidate_df["outlier_reason"] = reasons

    suspicious = candidate_df.sort_values("outlier_score", ascending=False).head(config.outlier_top_k)
    save_dataframe(candidate_df.sort_values("outlier_score", ascending=False), config.output_dir / "outlier_scores.csv")
    save_dataframe(suspicious, config.output_dir / "top_suspicious_samples.csv")

    create_image_montage(
        suspicious["image_path"].tolist(),
        config.output_dir / "top_suspicious_samples_montage.png",
        titles=[f"{score:.2f}" for score in suspicious["outlier_score"].tolist()],
        image_size=config.montage_image_size,
        ncols=config.montage_columns,
    )

    score_cols = ["outlier_quality_isolation", "outlier_embedding_nn", "outlier_class_centroid", "outlier_score"]
    score_frame = pd.DataFrame(np.nan, index=df.index, columns=score_cols)
    score_frame.loc[candidate_df.index, score_cols] = candidate_df[score_cols]
    reason_frame = pd.DataFrame({"outlier_reason": None}, index=df.index)
    reason_frame.loc[candidate_df.index, "outlier_reason"] = candidate_df["outlier_reason"]
    df = pd.concat([df, score_frame, reason_frame], axis=1)

    markdown_lines = [
        f"- Candidate images scored: `{len(candidate_df)}`",
        f"- Top suspicious samples exported: `{len(suspicious)}`",
        "- This stage is framed as prototype detection for non-class inputs, acquisition failures, and unusual examples.",
    ]
    for _, row in suspicious.head(8).iterrows():
        markdown_lines.append(
            f"- `{Path(row['image_path']).name}` score `{row['outlier_score']:.2f}` because {row['outlier_reason']}"
        )

    markdown = "\n".join(markdown_lines)
    write_text(config.output_dir / "outlier_summary.md", markdown + "\n")
    return df, {
        "title": "Outlier Detection",
        "markdown": markdown,
    }
