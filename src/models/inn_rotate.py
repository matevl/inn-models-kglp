import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.intervals import ComplexInterval, irotate


class RotatEEntityEmbedding(nn.Module):
    def __init__(self, num_entities: int, dim: int, init_rho: float = -5.0):
        super().__init__()
        self.center = nn.Embedding(num_entities, dim * 2)
        self.rho = nn.Embedding(num_entities, dim)

        nn.init.uniform_(self.center.weight, -0.1, 0.1)
        nn.init.constant_(self.rho.weight, init_rho)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.center(idx)
        r = F.softplus(self.rho(idx))
        return c, r


class INNRotatELinkPredictor(nn.Module):
    """Interval-based implementation of RotatE with isotropic geometry."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        gamma_margin: float = 1.0,
        init_rho: float = -5.0,
    ):
        super().__init__()
        self.dim = dim
        self.entity_emb = RotatEEntityEmbedding(num_entities, dim, init_rho=init_rho)
        self.rel_center = nn.Embedding(num_relations, dim)
        self.rel_rho = nn.Embedding(num_relations, dim)
        self.gamma_margin = gamma_margin
        self.init_rho = init_rho

        nn.init.uniform_(self.rel_center.weight, -math.pi, math.pi)
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
        """Compute the link prediction score for a batch of triples.

        Args:
            h_idx (torch.Tensor): Tensor of head entity indices.
            r_idx (torch.Tensor): Tensor of relation indices.
            t_idx (torch.Tensor): Tensor of tail entity indices.

        Returns:
            torch.Tensor: The computed scores for the input triples.
        """
        rc_phase, rr = self.get_relation(r_idx)

        hc, hr = self.entity_emb(h_idx)
        tc, tr = self.entity_emb(t_idx)

        hc_re, hc_im = torch.chunk(hc, 2, dim=-1)
        tc_re, tc_im = torch.chunk(tc, 2, dim=-1)

        h_interval = ComplexInterval(hc_re, hc_im, hr)
        t_interval = ComplexInterval(tc_re, tc_im, tr)

        pred_interval = irotate(h_interval, rc_phase, rr)

        return pred_interval.distance(t_interval)

    def forward(
        self, pos_triplets: torch.Tensor, neg_triplets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training with positive and negative triples.

        Args:
            pos_triplets (torch.Tensor): Tensor of shape (batch_size, 3) representing positive triples.
            neg_triplets (torch.Tensor): Tensor of shape (batch_size, num_negatives, 3) representing negative triples.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pos_scores (torch.Tensor): Scores for the positive triples.
                - neg_scores (torch.Tensor): Scores for the negative triples.
        """
        rc_phase, rr = self.get_relation(pos_triplets[:, 1])

        pos_h_idx = pos_triplets[:, 0]
        pos_t_idx = pos_triplets[:, 2]
        neg_h_idx = neg_triplets[:, :, 0]
        neg_t_idx = neg_triplets[:, :, 2]

        def compute_score(h_idx, t_idx, r_phase, r_r):
            hc, hr = self.entity_emb(h_idx)
            tc, tr = self.entity_emb(t_idx)

            hc_re, hc_im = torch.chunk(hc, 2, dim=-1)
            tc_re, tc_im = torch.chunk(tc, 2, dim=-1)

            h_interval = ComplexInterval(hc_re, hc_im, hr)
            t_interval = ComplexInterval(tc_re, tc_im, tr)

            pred_interval = irotate(h_interval, r_phase, r_r)

            return pred_interval.distance(t_interval)

        pos_scores = compute_score(pos_h_idx, pos_t_idx, rc_phase, rr)

        rc_phase_neg = rc_phase.unsqueeze(1)
        rr_neg = rr.unsqueeze(1)

        neg_scores = compute_score(
            neg_h_idx, neg_t_idx, rc_phase.unsqueeze(1), rr.unsqueeze(1)
        )

        return pos_scores, neg_scores

    def forward_1ton(self, pos_triplets: torch.Tensor) -> torch.Tensor:
        """1-to-N scoring against all entities."""
        num_ent = self.entity_emb.center.num_embeddings
        device = pos_triplets.device
        all_entity_ids = torch.arange(num_ent, device=device)
        u_c, u_r = self.entity_emb(all_entity_ids)

        h_idx = pos_triplets[:, 0]
        r_idx = pos_triplets[:, 1]

        hc, hr = self.entity_emb(h_idx)
        rc_phase, rr = self.get_relation(r_idx)

        hc_re, hc_im = torch.chunk(hc, 2, dim=-1)
        u_re, u_im = torch.chunk(u_c, 2, dim=-1)

        cos_r = torch.cos(rc_phase)
        sin_r = torch.sin(rc_phase)

        pred_re = hc_re * cos_r - hc_im * sin_r
        pred_im = hc_re * sin_r + hc_im * cos_r
        pred_r = hr + rr

        diff_re = pred_re.unsqueeze(1) - u_re.unsqueeze(0)
        diff_im = pred_im.unsqueeze(1) - u_im.unsqueeze(0)

        dist_c = torch.sqrt(diff_re**2 + diff_im**2).sum(dim=-1)
        sum_r = (pred_r.unsqueeze(1) + u_r.unsqueeze(0)).sum(dim=-1)

        return sum_r - dist_c
