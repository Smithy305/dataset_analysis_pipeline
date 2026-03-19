"""Build a CSV manifest for the Oxford-IIIT Pet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a CSV manifest for Oxford-IIIT Pet.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the Oxford-IIIT Pet dataset.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <dataset-root>/pets_manifest.csv.",
    )
    return parser.parse_args()


def breed_label_from_image_id(image_id: str) -> str:
    """Convert an Oxford Pets image id into a breed label."""

    return image_id.rsplit("_", 1)[0]


def species_from_code(species_code: str) -> str:
    """Map Oxford Pets species codes to labels."""

    return "cat" if str(species_code) == "1" else "dog"


def read_split_ids(path: Path) -> set[str]:
    """Read Oxford Pets split files and return image ids."""

    if not path.exists():
        return set()

    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ids.add(line.split()[0])
    return ids


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    """Read annotations/list.txt and create a clean manifest."""

    list_path = dataset_root / "annotations" / "list.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Could not find annotation file: {list_path}")

    trainval_ids = read_split_ids(dataset_root / "annotations" / "trainval.txt")
    test_ids = read_split_ids(dataset_root / "annotations" / "test.txt")

    rows: list[dict[str, object]] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        image_id, class_id, species_code, breed_id = line.split()
        rows.append(
            {
                "image_id": image_id,
                "image_path": f"{image_id}.jpg",
                "label": breed_label_from_image_id(image_id),
                "class_id": int(class_id),
                "species": species_from_code(species_code),
                "breed_id": int(breed_id),
                "split": "trainval" if image_id in trainval_ids else "test" if image_id in test_ids else "unknown",
                "is_trainval": image_id in trainval_ids,
                "is_test": image_id in test_ids,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else dataset_root / "pets_manifest.csv"

    manifest = build_manifest(dataset_root)
    manifest.to_csv(output_csv, index=False)
    print(f"Wrote {len(manifest)} rows to {output_csv}")


if __name__ == "__main__":
    main()
