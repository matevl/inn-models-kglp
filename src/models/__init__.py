from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .inn_ours_mlp import INNLinkPredictor
from .inn_lightgcn import INNLightGCNLinkPredictor
from .inn_compgcn import INNCompGCNLinkPredictor
from .inn_rotate import INNRotatELinkPredictor
from .inn_transe import INNTransELinkPredictor


def infer_model_type_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("entity_encoder.") for k in state_dict):
        return "inn_ours_mlp"
    if any(k.startswith("layer.W_in.") for k in state_dict):
        return "inn_compgcn"
    if any(k.startswith("entity_emb.") for k in state_dict) and not any(k.startswith("layer.") for k in state_dict) and "net.layers.0.W.weight" not in state_dict:
        # Check if it's rotate or transe
        if "rel_center.weight" in state_dict and state_dict["rel_center.weight"].shape[1] == state_dict["entity_emb.center.weight"].shape[1]:
            return "inn_transe"
        return "inn_rotate"
    if any(k.startswith("entity_emb.") for k in state_dict):
        return "inn_lightgcn"
    return "inn_ours_mlp"


def build_link_predictor(
    model_type: str,
    num_entities: int,
    num_relations: int,
    dim: int,
    gamma_margin: float,
    init_rho: float = -5.0,
    hidden_layers: list[int] | None = None,
) -> nn.Module:
    if model_type == "inn_ours_mlp":
        return INNLinkPredictor(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            gamma_margin=gamma_margin,
            init_rho=init_rho,
            hidden_layers=hidden_layers,
        )
    if model_type == "inn_lightgcn":
        return INNLightGCNLinkPredictor(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            gamma_margin=gamma_margin,
            init_rho=init_rho,
        )
    if model_type == "inn_compgcn":
        return INNCompGCNLinkPredictor(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            gamma_margin=gamma_margin,
            init_rho=init_rho,
        )
    if model_type == "inn_rotate":
        return INNRotatELinkPredictor(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            gamma_margin=gamma_margin,
            init_rho=init_rho,
        )
    if model_type == "inn_transe":
        return INNTransELinkPredictor(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            gamma_margin=gamma_margin,
            init_rho=init_rho,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def self_adversarial_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    gamma_margin: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    pos_loss = -F.logsigmoid(pos_scores + gamma_margin).mean()
    with torch.no_grad():
        neg_weights = F.softmax(neg_scores * alpha, dim=-1)
    neg_loss = -(neg_weights * F.logsigmoid(-neg_scores - gamma_margin)).sum(dim=-1).mean()
    return pos_loss + neg_loss


def bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    gamma_margin: float = 1.0,
    alpha: float = 1.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    pos_target = torch.full_like(pos_scores, 1.0 - label_smoothing)
    neg_target = torch.zeros_like(neg_scores)

    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_target)
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_target)
    return pos_loss + neg_loss


def compgcn_bce_loss(
    all_scores: torch.Tensor,
    target_indices: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Standard CompGCN loss: BCE with label smoothing for 1-to-N scoring."""
    num_entities = all_scores.size(1)
    
    targets = torch.full_like(all_scores, label_smoothing / num_entities)
    targets.scatter_(1, target_indices.unsqueeze(1), 1.0 - label_smoothing)
    
    return F.binary_cross_entropy_with_logits(all_scores, targets)


def margin_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    gamma_margin: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    loss = F.relu(gamma_margin - pos_scores.unsqueeze(1) + neg_scores)
    return loss.mean()


def logsumexp_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    gamma_margin: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    diff = neg_scores - pos_scores.unsqueeze(-1) + gamma_margin
    zero = torch.zeros_like(diff[:, :1])
    loss = torch.logsumexp(torch.cat([zero, diff], dim=-1), dim=-1)
    return loss.mean()


def softplus_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    gamma_margin: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    pos_loss = F.softplus(-pos_scores).mean()
    neg_loss = F.softplus(neg_scores).mean()
    return pos_loss + neg_loss


def infonce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    gamma_margin: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    # Treat as multi-class classification where alpha acts as inverse temperature (1/tau)
    scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) * alpha
    targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, targets)


LOSS_TYPE = {
    "self_adversarial": self_adversarial_loss,
    "bce": bce_loss,
    "gamma_margin": margin_ranking_loss,
    "logsumexp": logsumexp_loss,
    "softplus": softplus_loss,
    "infonce": infonce_loss,
}
