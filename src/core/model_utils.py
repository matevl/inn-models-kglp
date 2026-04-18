"""Model loading and resolution utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models import build_link_predictor, infer_model_type_from_state_dict
from utils.runtime import load_checkpoint


def _resolve_model_type(
    forced_model_type: str,
    checkpoint_config: dict,
    state_dict: dict[str, torch.Tensor],
) -> str:
    """Resolve model type from checkpoint or inference."""
    if forced_model_type != "auto":
        return forced_model_type

    configured = checkpoint_config.get("model_type")
    if configured in {"inn_ours_mlp", "inn_lightgcn", "transe", "rotate"}:
        return str(configured)

    return infer_model_type_from_state_dict(state_dict)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    default_dim: int,
    default_margin: float,
    forced_model_type: str = "auto",
    hidden_layer_size: int = 2,
) -> tuple[nn.Module, dict, str]:
    """Load model from checkpoint with configuration resolution."""
    checkpoint_data = load_checkpoint(checkpoint_path, device)
    cfg = checkpoint_data.get("config", {})
    dim = int(cfg.get("dim", default_dim))
    margin = float(cfg.get("margin", default_margin))
    state_dict = checkpoint_data["model_state_dict"]
    model_type = _resolve_model_type(forced_model_type, cfg, state_dict)

    loaded_hidden_coef = int(cfg.get("hidden_layer_size", hidden_layer_size))

    model = build_link_predictor(
        model_type=model_type,
        num_entities=int(checkpoint_data["num_entities"]),
        num_relations=int(checkpoint_data["num_relations"]),
        dim=dim,
        margin=margin,
        hidden_layer_size=loaded_hidden_coef,
    ).to(device)
    model.load_state_dict(state_dict)
    return model, checkpoint_data, model_type
