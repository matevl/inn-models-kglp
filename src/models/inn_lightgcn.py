from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .inn_ours_mlp import IntervalEntityEmbedding
from core.intervals import Interval, interval_relu


class IntervalGCNLayer(nn.Module):
    def __init__(self, i, o, act="relu"):
        super().__init__()
        self.W = nn.Linear(i, o, bias=False)
        self.act = act

    def forward(self, A, H: Interval):
        Zc = self.W(H.c)
        Zr = H.r @ self.W.weight.abs().t()

        # Disable autocast as PyTorch sparse matrix multiplication does not support FP16
        device_type = Zc.device.type
        with torch.autocast(
            device_type=device_type if device_type != "mps" else "cpu", enabled=False
        ):
            A_f32, Zc_f32, Zr_f32 = (
                A.to(torch.float32),
                Zc.to(torch.float32),
                Zr.to(torch.float32),
            )
            C_f32 = A_f32 @ Zc_f32
            R_f32 = A_f32.abs() @ Zr_f32

        C = C_f32.to(Zc.dtype)
        R = R_f32.to(Zr.dtype)

        Hn = Interval(C, R)
        if self.act == "relu":
            return interval_relu(Hn)
        return Hn


class IntervalLightGCN(nn.Module):
    def __init__(self, i, h, o, layers=2):
        super().__init__()
        dims = [i] + [h] * (layers - 1) + [o]
        self.layers = nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            self.layers.append(IntervalGCNLayer(a, b, act="relu"))

    def forward(self, A, H):
        for L in self.layers:
            H = L(A, H)
        return H


class INNLightGCNLinkPredictor(nn.Module):
    """LightGCN using static interval embeddings."""

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
        self.net = IntervalLightGCN(dim, max(dim, 64), dim, layers=2)
        self.gamma_margin = gamma_margin

        nn.init.uniform_(self.rel_center.weight, -0.1, 0.1)
        nn.init.constant_(self.rel_rho.weight, init_rho)
        self.register_buffer("A", None, persistent=False)

    def build_graph(self, train_triples: torch.Tensor) -> None:
        """Construct and store the normalized adjacency matrix."""
        num_ent = self.entity_emb.center.num_embeddings
        device = self.entity_emb.center.weight.device

        edges = train_triples[:, [0, 2]].t()  # Extract entity pairs (2 x E)
        edges = edges.to(device)
        edges = torch.cat([edges, edges[[1, 0]]], dim=1)  # Make graph undirected

        self_loops = torch.arange(num_ent, device=device).unsqueeze(0).repeat(2, 1)
        edges = torch.cat([edges, self_loops], dim=1)
        edges = torch.unique(edges, dim=1)

        row, col = edges
        deg = torch.bincount(row, minlength=num_ent).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
            A = torch.sparse_coo_tensor(edges, edge_weight, (num_ent, num_ent)).to(
                device
            )
        self.A = A

    def get_relation(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_r = self.rel_center(idx)
        r_r = F.softplus(self.rel_rho(idx))
        return c_r, r_r

    def compute_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute interval embeddings after GCN layers if adjacency matrix is available."""
        num_ent = self.entity_emb.center.num_embeddings
        device = self.entity_emb.center.weight.device
        all_entity_ids = torch.arange(num_ent, device=device)
        u_c, u_r = self.entity_emb(all_entity_ids)

        if self.A is not None:
            H = Interval(u_c, u_r)
            H = self.net(self.A, H)
            return H.c, H.r
        return u_c, u_r

    def inn_score(
        self,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        u_c, u_r = self.compute_all_embeddings()
        hc, hr = u_c[h_idx], u_r[h_idx]
        tc, tr = u_c[t_idx], u_r[t_idx]
        rc, rr = self.get_relation(r_idx)

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
        rc, rr = self.get_relation(pos_triplets[:, 1])

        u_c, u_r = self.compute_all_embeddings()

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

    def forward_1ton(self, pos_triplets: torch.Tensor) -> torch.Tensor:
        """1-to-N scoring against all entities."""
        u_c, u_r = self.compute_all_embeddings()
        rc, rr = self.get_relation(pos_triplets[:, 1])

        h_idx = pos_triplets[:, 0]
        hc, hr = u_c[h_idx], u_r[h_idx]

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
