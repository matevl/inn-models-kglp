from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from .inn_ours_mlp import IntervalEntityEmbedding


class INNLightGCNLinkPredictor(nn.Module):
    """LightGCN using static interval embeddings."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        margin: float = 1.0,
        init_rho: float = -5.0,
    ):
        super().__init__()
        self.entity_emb = IntervalEntityEmbedding(num_entities, dim, init_rho=init_rho)
        self.rel_center = nn.Embedding(num_relations, dim)
        self.rel_rho = nn.Embedding(num_relations, dim)
        self.margin = margin

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
        max_radius_sum = torch.norm(pred_r + tr, p=1, dim=-1)
        return max_radius_sum - distance

    def forward(
        self,
        pos_triplets: torch.Tensor,
        neg_triplets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute relation embeddings once (same for pos and neg)
        rc, rr = self.get_relation(pos_triplets[:, 1])

        # Compute embeddings for all entities rather than isolating unique ones
        num_ent = self.entity_emb.center.num_embeddings
        all_entity_ids = torch.arange(num_ent, device=pos_triplets.device)
        u_c, u_r = self.entity_emb(all_entity_ids)

        pos_h_idx = pos_triplets[:, 0]
        pos_t_idx = pos_triplets[:, 2]
        neg_h_idx = neg_triplets[:, :, 0]
        neg_t_idx = neg_triplets[:, :, 2]

        hc = u_c[pos_h_idx]
        hr = u_r[pos_h_idx]
        tc = u_c[pos_t_idx]
        tr = u_r[pos_t_idx]

        pred_c = hc + rc
        pred_r = hr + rr

        distance = torch.norm(pred_c - tc, p=1, dim=-1)
        max_radius_sum = torch.norm(pred_r + tr, p=1, dim=-1)
        pos_scores = max_radius_sum - distance

        hc_neg = u_c[neg_h_idx]
        hr_neg = u_r[neg_h_idx]
        tc_neg = u_c[neg_t_idx]
        tr_neg = u_r[neg_t_idx]

        rc_neg = rc.unsqueeze(1)
        rr_neg = rr.unsqueeze(1)

        pred_c_neg = hc_neg + rc_neg
        pred_r_neg = hr_neg + rr_neg

        distance_neg = torch.norm(pred_c_neg - tc_neg, p=1, dim=-1)
        max_radius_sum_neg = torch.norm(pred_r_neg + tr_neg, p=1, dim=-1)
        neg_scores = max_radius_sum_neg - distance_neg

        return pos_scores, neg_scores

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
