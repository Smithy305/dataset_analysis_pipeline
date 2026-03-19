"""Placeholder extension point for future multi-view and coverage analysis."""

from __future__ import annotations

from pathlib import Path

from src.utils import write_text


def write_multiview_placeholder(output_dir: Path) -> dict[str, str]:
    """Write a small design note for future multi-view work."""

    text = """This project intentionally leaves multi-view analysis as a separate extension point.

Possible future additions:

- group images by specimen, donor, or case identifier
- infer whether required viewpoints are present
- score viewpoint coverage completeness
- compare single-view versus multi-view embeddings
- attach sparse geometry or Gaussian-splatting-inspired reconstructions for richer visual coverage analysis

Suggested integration point:

- add a case-level manifest with columns such as `case_id`, `view_id`, and `capture_order`
- build a case-centric analysis module that consumes the per-image quality and embedding outputs already generated here
"""
    write_text(output_dir / "multiview_placeholder.md", text)
    return {
        "title": "Multi-view Placeholder",
        "markdown": text,
    }
