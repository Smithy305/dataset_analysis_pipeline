"""Near-duplicate and perceptual-similarity analysis."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import AnalysisConfig
from src.utils import safe_open_image, save_dataframe, write_text
from src.visualizations import create_image_montage


def _average_hash(image: Image.Image, size: int = 8) -> np.ndarray:
    gray = image.convert("L").resize((size, size))
    arr = np.asarray(gray, dtype=np.float32)
    return (arr >= arr.mean()).astype(np.uint8).reshape(-1)


def _difference_hash(image: Image.Image, size: int = 8) -> np.ndarray:
    gray = image.convert("L").resize((size + 1, size))
    arr = np.asarray(gray, dtype=np.float32)
    return (arr[:, 1:] >= arr[:, :-1]).astype(np.uint8).reshape(-1)


def _hash_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.tolist():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _hamming_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    return int(np.count_nonzero(bits_a != bits_b))


def run_duplicate_detection(df: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect strong duplicate candidates and near-duplicates using perceptual hashes."""

    readable_df = df[df["is_readable"]].copy()
    if readable_df.empty:
        return df, {
            "title": "Duplicate Detection",
            "markdown": "No readable images were available for duplicate analysis.",
        }

    ahash_bits: dict[int, np.ndarray] = {}
    dhash_bits: dict[int, np.ndarray] = {}
    ahash_hex: dict[int, str] = {}
    dhash_hex: dict[int, str] = {}

    for idx, row in tqdm(readable_df.iterrows(), total=len(readable_df), desc="Computing perceptual hashes"):
        image, _ = safe_open_image(Path(row["image_path"]))
        if image is None:
            continue
        a_bits = _average_hash(image)
        d_bits = _difference_hash(image)
        ahash_bits[idx] = a_bits
        dhash_bits[idx] = d_bits
        ahash_hex[idx] = _hash_to_hex(a_bits)
        dhash_hex[idx] = _hash_to_hex(d_bits)

    hash_frame = pd.DataFrame(index=df.index, data={"ahash": None, "dhash": None})
    for idx, value in ahash_hex.items():
        hash_frame.loc[idx, "ahash"] = value
    for idx, value in dhash_hex.items():
        hash_frame.loc[idx, "dhash"] = value
    df = pd.concat([df, hash_frame], axis=1)

    exact_groups = (
        pd.Series(ahash_hex, name="ahash")
        .reset_index()
        .rename(columns={"index": "row_index"})
        .groupby("ahash")["row_index"]
        .apply(list)
    )
    exact_groups = exact_groups[exact_groups.map(len) > 1]

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, hash_value in ahash_hex.items():
        buckets[hash_value[:4]].append(idx)

    near_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        for left_idx, right_idx in combinations(bucket_indices, 2):
            pair = (min(left_idx, right_idx), max(left_idx, right_idx))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            a_dist = _hamming_distance(ahash_bits[left_idx], ahash_bits[right_idx])
            d_dist = _hamming_distance(dhash_bits[left_idx], dhash_bits[right_idx])
            if a_dist <= config.duplicate_hamming_threshold and d_dist <= config.duplicate_hamming_threshold:
                near_pairs.append(
                    {
                        "left_index": pair[0],
                        "right_index": pair[1],
                        "left_image_path": df.loc[pair[0], "image_path"],
                        "right_image_path": df.loc[pair[1], "image_path"],
                        "left_label": df.loc[pair[0], "label"],
                        "right_label": df.loc[pair[1], "label"],
                        "ahash_distance": a_dist,
                        "dhash_distance": d_dist,
                    }
                )

    near_pairs_df = pd.DataFrame(near_pairs).sort_values(
        ["ahash_distance", "dhash_distance", "left_label", "right_label"],
        ignore_index=True,
    ) if near_pairs else pd.DataFrame(
        columns=[
            "left_index",
            "right_index",
            "left_image_path",
            "right_image_path",
            "left_label",
            "right_label",
            "ahash_distance",
            "dhash_distance",
        ]
    )

    exact_rows: list[dict[str, Any]] = []
    for cluster_id, (hash_value, indices) in enumerate(exact_groups.items(), start=1):
        for idx in indices:
            exact_rows.append(
                {
                    "cluster_id": cluster_id,
                    "ahash": hash_value,
                    "image_path": df.loc[idx, "image_path"],
                    "label": df.loc[idx, "label"],
                }
            )
    exact_df = pd.DataFrame(exact_rows)

    save_dataframe(exact_df, config.output_dir / "exact_hash_match_groups.csv")
    save_dataframe(near_pairs_df, config.output_dir / "near_duplicate_pairs.csv")

    montage_paths: list[str] = []
    montage_titles: list[str] = []
    for _, row in near_pairs_df.head(config.duplicate_top_k).iterrows():
        montage_paths.extend([row["left_image_path"], row["right_image_path"]])
        montage_titles.extend(
            [
                f"{row['left_label']} a={row['ahash_distance']} d={row['dhash_distance']}",
                f"{row['right_label']} a={row['ahash_distance']} d={row['dhash_distance']}",
            ]
        )
    create_image_montage(
        montage_paths,
        config.output_dir / "near_duplicate_pairs_montage.png",
        titles=montage_titles,
        image_size=config.montage_image_size,
        ncols=2,
    )

    markdown_lines = [
        f"- Exact hash-match groups: `{exact_df['cluster_id'].nunique() if not exact_df.empty else 0}`",
        f"- Near-duplicate pairs within threshold: `{len(near_pairs_df)}`",
        f"- Hamming threshold per hash: `{config.duplicate_hamming_threshold}`",
    ]
    if len(near_pairs_df):
        for _, row in near_pairs_df.head(8).iterrows():
            markdown_lines.append(
                f"- `{Path(row['left_image_path']).name}` vs `{Path(row['right_image_path']).name}` "
                f"with ahash `{row['ahash_distance']}` and dhash `{row['dhash_distance']}`"
            )

    markdown = "\n".join(markdown_lines)
    write_text(config.output_dir / "duplicate_detection.md", markdown + "\n")
    return df, {
        "title": "Duplicate Detection",
        "markdown": markdown,
    }
