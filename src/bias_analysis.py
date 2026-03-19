"""Bias and shortcut-learning oriented dataset diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import AnalysisConfig
from src.utils import save_dataframe, write_text


def _cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return 0.0
    mean_diff = sample_a.mean() - sample_b.mean()
    pooled_var = ((len(sample_a) - 1) * sample_a.var(ddof=1) + (len(sample_b) - 1) * sample_b.var(ddof=1))
    pooled_var /= max(len(sample_a) + len(sample_b) - 2, 1)
    pooled_std = np.sqrt(max(pooled_var, 1e-8))
    return float(mean_diff / pooled_std)


def _effect_table(df: pd.DataFrame, group_col: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean_df = df.dropna(subset=[group_col]).copy()
    for metric in metrics:
        metric_df = clean_df[[group_col, metric]].dropna()
        if metric_df.empty:
            continue
        for group_name, group_values in metric_df.groupby(group_col):
            in_group = group_values[metric].to_numpy()
            out_group = metric_df.loc[metric_df[group_col] != group_name, metric].to_numpy()
            effect = _cohens_d(in_group, out_group) if len(out_group) else 0.0
            rows.append(
                {
                    "group_type": group_col,
                    "group_name": group_name,
                    "metric": metric,
                    "group_count": len(in_group),
                    "mean_in_group": float(np.mean(in_group)),
                    "mean_out_group": float(np.mean(out_group)) if len(out_group) else np.nan,
                    "cohens_d_vs_rest": effect,
                    "suspicious_shortcut_signal": abs(effect) >= 0.8 and len(in_group) >= 10,
                }
            )
    return pd.DataFrame(rows)


def _domain_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    if "analysis_domain" not in df.columns or df["analysis_domain"].dropna().empty:
        return pd.DataFrame()
    balance = pd.crosstab(df["label"], df["analysis_domain"])
    return balance.reset_index()


def run_bias_analysis(df: pd.DataFrame, config: AnalysisConfig) -> dict[str, Any]:
    """Analyse class imbalance and acquisition-driven shortcut risk."""

    readable_df = df[df["is_readable"]].copy()
    class_counts = readable_df["label"].value_counts().rename_axis("label").reset_index(name="count")
    class_counts["fraction"] = class_counts["count"] / max(class_counts["count"].sum(), 1)
    save_dataframe(class_counts, config.output_dir / "bias_class_balance.csv")

    class_effects = _effect_table(readable_df, "label", config.quality_metric_columns)
    save_dataframe(class_effects, config.output_dir / "bias_quality_effects_by_class.csv")

    domain_effects = pd.DataFrame()
    if "analysis_domain" in readable_df.columns and readable_df["analysis_domain"].nunique(dropna=True) > 1:
        domain_effects = _effect_table(readable_df, "analysis_domain", config.quality_metric_columns)
        save_dataframe(domain_effects, config.output_dir / "bias_quality_effects_by_domain.csv")

    domain_balance = _domain_balance_table(readable_df)
    if not domain_balance.empty:
        save_dataframe(domain_balance, config.output_dir / "label_by_analysis_domain.csv")

    suspicious = pd.concat([class_effects, domain_effects], ignore_index=True) if not domain_effects.empty else class_effects
    suspicious = suspicious[suspicious["suspicious_shortcut_signal"]].sort_values("cohens_d_vs_rest", key=lambda s: s.abs(), ascending=False)
    save_dataframe(suspicious, config.output_dir / "suspicious_shortcut_signals.csv")

    bullets = [
        f"- Class imbalance range: `{class_counts['count'].min()}` to `{class_counts['count'].max()}` images per class",
        f"- Suspicious quality-vs-group signals: `{len(suspicious)}`",
    ]
    if not domain_balance.empty:
        domain_kind = readable_df["analysis_domain_kind"].iloc[0] if "analysis_domain_kind" in readable_df.columns else "unknown"
        bullets.append(
            f"- Analysis-domain grouping used `{domain_kind}` and included acquisition-proxy effects in the spurious-correlation analysis."
        )
    else:
        bullets.append("- No useful analysis-domain grouping was available; only class-conditioned quality effects were analysed.")

    if len(suspicious):
        top_rows = suspicious.head(8)
        bullets.append("")
        bullets.append("Potential shortcut signals:")
        for _, row in top_rows.iterrows():
            bullets.append(
                f"- `{row['group_type']}={row['group_name']}` differs on `{row['metric']}` with Cohen's d `{row['cohens_d_vs_rest']:.2f}`"
            )

    markdown = "\n".join(bullets)
    write_text(config.output_dir / "bias_analysis.md", markdown + "\n")
    return {
        "title": "Bias And Shortcut Learning Analysis",
        "markdown": markdown,
    }
