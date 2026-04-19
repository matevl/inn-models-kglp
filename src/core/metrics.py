"""Metrics formatting utilities."""

from __future__ import annotations

from typing import Dict


def format_metrics_table(
    metrics: Dict[str, float], title: str = "Evaluation Metrics"
) -> str:
    """
    Format evaluation metrics as a nicely formatted table.

    Args:
            metrics: Dictionary of metric name -> value
            title: Title for the metrics table

    Returns:
            Formatted string representation
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append("=" * 60)

    for metric_name, metric_value in metrics.items():
        # Format metric name nicely (mrr -> MRR, hits_at_1 -> Hits@1, etc.)
        if metric_name == "mrr":
            display_name = "MRR (Mean Reciprocal Rank)"
        elif metric_name.startswith("hits_at_"):
            k = metric_name.split("_")[-1]
            display_name = f"Hits@{k}"
        else:
            display_name = metric_name.replace("_", " ").title()

        lines.append(f"  {display_name:<40} {metric_value:>15.6f}")

    lines.append("=" * 60)
    return "\n".join(lines)
