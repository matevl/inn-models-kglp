from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.intervals import Interval, interval_relu


class IntervalEntityEmbedding(nn.Module):
    def __init__(self, num_entities: int, embedding_dim: int, init_rho: float = -5.0):
        super().__init__()
        self.center = nn.Embedding(num_entities, embedding_dim)
        self.rho = nn.Embedding(num_entities, embedding_dim)

        nn.init.uniform_(self.center.weight, -0.1, 0.1)
        nn.init.constant_(self.rho.weight, init_rho)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.center(idx)
        r = F.softplus(self.rho(idx))
        return c, r


class IntervalLinear_INN_Ours_MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight_c = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.weight_r = nn.Parameter(torch.full((out_dim, in_dim), -5.0))
        self.bias_c = nn.Parameter(torch.zeros(out_dim))
        self.bias_r = nn.Parameter(torch.full((out_dim,), -5.0))

    def forward(
        self, center: torch.Tensor, radius: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_center = self.weight_c
        w_radius = F.softplus(self.weight_r)
        b_radius = F.softplus(self.bias_r)

        center_out = center @ w_center.t() + self.bias_c
        w_combined = w_center.abs() + w_radius
        radius_out = (
            (center.abs() @ w_radius.t()) + (radius @ w_combined.t()) + b_radius
        )
        return center_out, radius_out


class IntervalMLP_INN_Ours_MLP(nn.Module):
    def __init__(
        self,
        num_entities: int,
        dim: int,
        hidden_layers: list[int] | None = None,
        init_rho: float = -5.0,
    ):
        super().__init__()

        self.emb_c = nn.Embedding(num_entities, dim)
        self.emb_r = nn.Embedding(num_entities, dim)

        hidden_layers = hidden_layers or []
        self.mlp_layers = nn.ModuleList()

        in_dim = dim
        for h_dim in hidden_layers:
            self.mlp_layers.append(IntervalLinear_INN_Ours_MLP(in_dim, h_dim))
            in_dim = h_dim

        # Output layer maps back to standard embedding dimension
        self.output_layer = IntervalLinear_INN_Ours_MLP(in_dim, dim)

        nn.init.uniform_(self.emb_c.weight, -0.1, 0.1)
        nn.init.constant_(self.emb_r.weight, init_rho)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.emb_c(idx)
        r = F.softplus(self.emb_r(idx))

        for layer in self.mlp_layers:
            c, r = layer(c, r)
            i_relu = interval_relu(Interval(c, r))
            c, r = i_relu.c, i_relu.r

        c, r = self.output_layer(c, r)

        return c, r


class INNLinkPredictor(nn.Module):
    """Default link predictor based on INN_Ours_MLP-INN entity encoder."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        hidden_layers: list[int] | None = None,
        margin: float = 1.0,
        init_rho: float = -5.0,
    ):
        super().__init__()
        self.entity_encoder = IntervalMLP_INN_Ours_MLP(
            num_entities, dim, hidden_layers=hidden_layers, init_rho=init_rho
        )
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
        hc, hr = self.entity_encoder(h_idx)
        rc, rr = self.get_relation(r_idx)
        tc, tr = self.entity_encoder(t_idx)

        pred_c = hc + rc
        pred_r = hr + rr

        distance = torch.norm(pred_c - tc, p=1, dim=-1)
        max_radius_sum = torch.norm(pred_r + tr, p=1, dim=-1)
        score = max_radius_sum - distance
        return score

    def forward(
        self,
        pos_triplets: torch.Tensor,
        neg_triplets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute relation embeddings once (same for pos and neg)
        rc, rr = self.get_relation(pos_triplets[:, 1])

        # Extract unique entities
        all_entities = torch.cat(
            [
                pos_triplets[:, 0],
                pos_triplets[:, 2],
                neg_triplets[:, :, 0].flatten(),
                neg_triplets[:, :, 2].flatten(),
            ]
        )
        unique_entities, inverse_indices = torch.unique(
            all_entities, return_inverse=True
        )

        # Compute embeddings only for unique entities
        u_c, u_r = self.entity_encoder(unique_entities)

        # Map back embeddings
        B = pos_triplets.size(0)
        num_neg = neg_triplets.size(1)

        pos_h_idx = inverse_indices[:B]
        pos_t_idx = inverse_indices[B : B * 2]
        neg_h_idx = inverse_indices[B * 2 : B * 2 + B * num_neg].view(B, num_neg)
        neg_t_idx = inverse_indices[B * 2 + B * num_neg :].view(B, num_neg)

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
            e_r = F.softplus(self.entity_encoder.emb_r.weight)
            rel_r = F.softplus(self.rel_rho.weight)
            return {
                "entity_r_mean": e_r.mean().item(),
                "entity_r_max": e_r.max().item(),
                "rel_r_mean": rel_r.mean().item(),
                "rel_r_max": rel_r.max().item(),
            }
