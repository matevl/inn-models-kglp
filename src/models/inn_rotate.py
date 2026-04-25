import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def __init__(self, num_entities: int, num_relations: int, dim: int, gamma_margin: float = 1.0, init_rho: float = -5.0):
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

        rc_re = torch.cos(rc_phase)
        rc_im = torch.sin(rc_phase)
        
        hc, hr = self.entity_emb(h_idx)
        tc, tr = self.entity_emb(t_idx)

        hc_re, hc_im = torch.chunk(hc, 2, dim=-1)
        tc_re, tc_im = torch.chunk(tc, 2, dim=-1)

        pred_c_re = hc_re * rc_re - hc_im * rc_im
        pred_c_im = hc_re * rc_im + hc_im * rc_re
        
        pred_r = hr + rr

        diff_c_re = pred_c_re - tc_re
        diff_c_im = pred_c_im - tc_im
        
        dist_c = torch.sqrt(diff_c_re**2 + diff_c_im**2).sum(dim=-1)
        sum_r = (pred_r + tr).sum(dim=-1)

        return sum_r - dist_c

    def forward(self, pos_triplets: torch.Tensor, neg_triplets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        rc_re = torch.cos(rc_phase)
        rc_im = torch.sin(rc_phase)

        pos_h_idx = pos_triplets[:, 0]
        pos_t_idx = pos_triplets[:, 2]
        neg_h_idx = neg_triplets[:, :, 0]
        neg_t_idx = neg_triplets[:, :, 2]

        def compute_score(h_idx, t_idx, rc_re, rc_im, rr):
            hc, hr = self.entity_emb(h_idx)
            tc, tr = self.entity_emb(t_idx)

            hc_re, hc_im = torch.chunk(hc, 2, dim=-1)
            tc_re, tc_im = torch.chunk(tc, 2, dim=-1)

            pred_c_re = hc_re * rc_re - hc_im * rc_im
            pred_c_im = hc_re * rc_im + hc_im * rc_re
            
            pred_r = hr + rr

            diff_c_re = pred_c_re - tc_re
            diff_c_im = pred_c_im - tc_im
            
            dist_c = torch.sqrt(diff_c_re**2 + diff_c_im**2).sum(dim=-1)
            sum_r = (pred_r + tr).sum(dim=-1)

            return sum_r - dist_c

        pos_scores = compute_score(pos_h_idx, pos_t_idx, rc_re, rc_im, rr)
        
        rc_re_neg = rc_re.unsqueeze(1)
        rc_im_neg = rc_im.unsqueeze(1)
        rr_neg = rr.unsqueeze(1)
        
        neg_scores = compute_score(neg_h_idx, neg_t_idx, rc_re_neg, rc_im_neg, rr_neg)

        return pos_scores, neg_scores
