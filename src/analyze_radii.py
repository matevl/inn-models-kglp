from __future__ import annotations

"""
Radius (Ray) Analysis Tool for Interval Neural Networks.

This script extracts and analyzes the geometry (radii/rays) of trained INN models.
It supports multiple architectures (GCN, MLP, RotatE, TransE) and provides
statistics and distributions for both base embeddings and transformed intervals.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

# Ensure we can import from the src directory
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root / "src"))

from data.dataset import load_dataset
from models import build_link_predictor, infer_model_type_from_state_dict
from utils.runtime import load_checkpoint, select_device, configure_logging

LOGGER = logging.getLogger("inn-models-kglp.analysis")


def get_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute descriptive statistics for a given tensor.

    Args:
        tensor: Input torch.Tensor (e.g., entity radii).

    Returns:
        A dictionary containing mean, std, min, max, and percentiles.
    """
    if tensor.numel() == 0:
        return {}

    vals = tensor.detach().cpu().numpy()
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def plot_distribution(vals: torch.Tensor, title: str, save_path: Path):
    """
    Generate and save a histogram of value distributions.

    Args:
        vals: Tensor containing the values to plot.
        title: Title of the histogram.
        save_path: Path where the resulting image will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        vals.detach().cpu().numpy().flatten(),
        bins=100,
        color="#4A90E2",
        edgecolor="black",
        alpha=0.7,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Radius Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_radii(
    checkpoint_path: str, dataset_path: str | None, output_dir: str, device_name: str
):
    """
    Main analysis pipeline: load model, extract radii, compute stats, and generate plots.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        dataset_path: Optional path to dataset (required for GCN-based models).
        output_dir: Directory where results will be stored.
        device_name: Hardware device (cuda, cpu, auto).
    """
    device = select_device(device_name)
    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"[ACTION] Loading checkpoint from {ckpt_path}")
    checkpoint_data = load_checkpoint(ckpt_path, device)
    state_dict = checkpoint_data["model_state_dict"]

    # Clean state_dict keys if they come from torch.compile (_orig_mod. prefix)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Identify model architecture from weight structure
    model_type = infer_model_type_from_state_dict(state_dict)
    config = checkpoint_data.get("config", {})
    num_entities = checkpoint_data.get("num_entities")
    num_relations = checkpoint_data.get("num_relations")

    LOGGER.info(f"[ACTION] Detected model type: {model_type}")

    # Reconstruct model architecture
    model = build_link_predictor(
        model_type=model_type,
        num_entities=num_entities,
        num_relations=num_relations,
        dim=config.get("dim", 128),
        gamma_margin=config.get("gamma_margin", 1.0),
        init_rho=config.get("init_rho", -5.0),
        hidden_layers=config.get("hidden_layers", []),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    results = {
        "model_type": model_type,
        "checkpoint": str(ckpt_path),
        "entities": {},
        "relations": {},
    }

    # 1. Base Radii Analysis (embeddings before any transformation)
    with torch.no_grad():
        e_rho = None
        if hasattr(model, "entity_emb") and hasattr(model.entity_emb, "rho"):
            e_rho = model.entity_emb.rho.weight
        elif hasattr(model, "entity_encoder") and hasattr(
            model.entity_encoder, "emb_r"
        ):
            e_rho = model.entity_encoder.emb_r.weight

        if e_rho is not None:
            e_radii_base = F.softplus(e_rho)
            results["entities"]["base"] = get_stats(e_radii_base)
            plot_distribution(
                e_radii_base,
                f"Entity Base Radii - {model_type}",
                out_dir / "entity_radii_base.png",
            )

        if hasattr(model, "rel_rho"):
            r_rho = model.rel_rho.weight
            r_radii_base = F.softplus(r_rho)
            results["relations"]["base"] = get_stats(r_radii_base)
            plot_distribution(
                r_radii_base,
                f"Relation Base Radii - {model_type}",
                out_dir / "relation_radii_base.png",
            )

    # 2. Propagated Radii Analysis (output of encoder layers)
    if dataset_path or model_type not in ["inn_lightgcn", "inn_compgcn"]:
        try:
            with torch.no_grad():
                if model_type == "inn_ours_mlp":
                    # For MLP models, pass all entity IDs through the encoder
                    all_ids = torch.arange(num_entities, device=device)
                    _, e_radii_prop = model.entity_encoder(all_ids)
                    results["entities"]["propagated"] = get_stats(e_radii_prop)
                    plot_distribution(
                        e_radii_prop,
                        f"Entity Propagated Radii - {model_type}",
                        out_dir / "entity_radii_propagated.png",
                    )

                elif model_type in ["inn_lightgcn", "inn_compgcn"]:
                    if dataset_path:
                        LOGGER.info(
                            f"[ACTION] Building graph from dataset: {dataset_path}"
                        )
                        dataset = load_dataset(dataset_path)
                        model.build_graph(dataset.train)

                        if model_type == "inn_lightgcn":
                            _, e_radii_prop = model.compute_all_embeddings()
                            results["entities"]["propagated"] = get_stats(e_radii_prop)
                            plot_distribution(
                                e_radii_prop,
                                f"Entity Propagated Radii - {model_type}",
                                out_dir / "entity_radii_propagated.png",
                            )
                        else:  # CompGCN
                            _, e_radii_prop, _, r_radii_prop = (
                                model.compute_all_embeddings()
                            )
                            results["entities"]["propagated"] = get_stats(e_radii_prop)
                            results["relations"]["propagated"] = get_stats(r_radii_prop)
                            plot_distribution(
                                e_radii_prop,
                                f"Entity Propagated Radii - {model_type}",
                                out_dir / "entity_radii_propagated.png",
                            )
                            plot_distribution(
                                r_radii_prop,
                                f"Relation Propagated Radii - {model_type}",
                                out_dir / "relation_radii_propagated.png",
                            )
                    else:
                        LOGGER.warning(
                            "[SKIP] Propagated analysis skipped: Dataset required for GCN architectures."
                        )
        except Exception as e:
            LOGGER.error(f"[ERROR] Propagated radii computation failed: {e}")

    # 3. Structural Analysis: Identify extreme geometry (Outliers)
    if e_rho is not None:
        # Calculate mean radius per entity across all dimensions
        e_radii = F.softplus(e_rho).mean(dim=-1)
        top_large = torch.topk(e_radii, min(10, num_entities)).indices.cpu().tolist()
        top_small = (
            torch.topk(e_radii, min(10, num_entities), largest=False)
            .indices.cpu()
            .tolist()
        )
        results["entities"]["top_largest_ids"] = top_large
        results["entities"]["top_smallest_ids"] = top_small

    # Final report generation
    report_path = out_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    LOGGER.info(f"[RECAP] Analysis successful. Results saved to: {out_dir}")
    LOGGER.info(
        f"[RECAP] Entity Base Radius Mean: {results['entities'].get('base', {}).get('mean', 'N/A'):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interval Geometry Analysis: Study radius (ray) distributions in trained INN models."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model (.pt)"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset path (required for graph-based models)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Target directory for output",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Execution device (cuda/cpu/auto)"
    )

    args = parser.parse_args()

    configure_logging()
    analyze_radii(args.checkpoint, args.dataset, args.output_dir, args.device)


if __name__ == "__main__":
    main()
