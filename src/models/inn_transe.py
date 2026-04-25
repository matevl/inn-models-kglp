from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.intervals import Interval
from .inn_ours_mlp import IntervalEntityEmbedding


class INNTransELinkPredictor(nn.Module):
    """Simple interval-based TransE implementation."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        gamma_margin: float = 1.0,
        init_rho: float = -5.0,
    ):
        super().__init__()
        self.entity_emb = IntervalEntityEmbedding(num_entities, dim, init_rho=init_rho)
        self.rel_center = nn.Embedding(num_relations, dim)
        self.rel_rho = nn.Embedding(num_relations, dim)
        self.gamma_margin = gamma_margin

        nn.init.uniform_(self.rel_center.weight, -0.1, 0.1)
        nn.init.constant_(self.rel_rho.weight, init_rho)

    def get_relation(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_r = self.rel_center(idx)
        r_r = F.softplus(self.rel_rho(idx))
        return c_r, r_r

    def inn_score(
        self,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        hc, hr = self.entity_emb(h_idx)
        rc, rr = self.get_relation(r_idx)
        tc, tr = self.entity_emb(t_idx)

        pred_c = hc + rc
        pred_r = hr + rr

        distance = torch.norm(pred_c - tc, p=1, dim=-1)
        max_radius_sum = (pred_r + tr).sum(dim=-1)
        return max_radius_sum - distance

    def forward(
        self,
        pos_triplets: torch.Tensor,
        neg_triplets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rc, rr = self.get_relation(pos_triplets[:, 1])

        pos_h_idx = pos_triplets[:, 0]
        pos_t_idx = pos_triplets[:, 2]
        neg_h_idx = neg_triplets[:, :, 0]
        neg_t_idx = neg_triplets[:, :, 2]

        # Positive scores
        hc, hr = self.entity_emb(pos_h_idx)
        tc, tr = self.entity_emb(pos_t_idx)

        pred_c = hc + rc
        pred_r = hr + rr

        distance = torch.norm(pred_c - tc, p=1, dim=-1)
        max_radius_sum = (pred_r + tr).sum(dim=-1)
        pos_scores = max_radius_sum - distance

        # Negative scores
        hc_neg, hr_neg = self.entity_emb(neg_h_idx)
        tc_neg, tr_neg = self.entity_emb(neg_t_idx)
        
        rc_neg = rc.unsqueeze(1)
        rr_neg = rr.unsqueeze(1)

        pred_c_neg = hc_neg + rc_neg
        pred_r_neg = hr_neg + rr_neg

        distance_neg = torch.norm(pred_c_neg - tc_neg, p=1, dim=-1)
        max_radius_sum_neg = (pred_r_neg + tr_neg).sum(dim=-1)
        neg_scores = max_radius_sum_neg - distance_neg

        return pos_scores, neg_scores

    def forward_1ton(self, pos_triplets: torch.Tensor) -> torch.Tensor:
        """1-to-N scoring against all entities."""
        num_ent = self.entity_emb.center.num_embeddings
        all_entity_ids = torch.arange(num_ent, device=pos_triplets.device)
        u_c, u_r = self.entity_emb(all_entity_ids)
        
        h_idx = pos_triplets[:, 0]
        r_idx = pos_triplets[:, 1]
        
        hc, hr = u_c[h_idx], u_r[h_idx]
        rc, rr = self.get_relation(r_idx)
        
        pred_c = hc + rc
        pred_r = hr + rr
        
        diff_c = pred_c.unsqueeze(1) - u_c.unsqueeze(0)
        distance = torch.norm(diff_c, p=1, dim=-1)
        
        sum_r = pred_r.unsqueeze(1) + u_r.unsqueeze(0)
        max_radius_sum = sum_r.sum(dim=-1)
        
        return max_radius_sum - distance

    def get_radii_stats(self) -> dict[str, float]:
        with torch.no_grad():
            e_r = F.softplus(self.entity_emb.rho.weight)
            rel_r = F.softplus(self.rel_rho.weight)
            return {
                "entity_r_mean": e_r.mean().item(),
                "entity_r_max": e_r.max().item(),
                "rel_r_mean": rel_r.mean().item(),
                "rel_r_max": rel_r.max().item(),
            }
