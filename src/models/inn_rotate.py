import torch
import torch.nn as nn
import torch.nn.functional as F

from .inn_ours_mlp import IntervalEntityEmbedding
import math

class INNRotatELinkPredictor(nn.Module):
    """Interval-based implementation of RotatE."""
    
    def __init__(self, num_entities: int, num_relations: int, dim: int, margin: float = 1.0, init_rho: float = -5.0):
        super().__init__()
        # For RotatE, we use double entity embedding dimension to represent complex numbers,
        # but in our INN logic we can just process real and imaginary parts explicitly.
        self.dim = dim
        self.entity_emb = IntervalEntityEmbedding(num_entities, dim * 2, init_rho=init_rho)
        self.rel_center = nn.Embedding(num_relations, dim)
        self.rel_rho = nn.Embedding(num_relations, dim)
        self.margin = margin
        self.init_rho = init_rho
        
        # Ensure relations' phases are mostly uniform initially
        self.embedding_range = (self.margin + 2.0) / dim
        nn.init.uniform_(self.rel_center.weight, -self.embedding_range, self.embedding_range)
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
        pi = math.pi
        
        # Relations only have phase, so we project them to phase angles
        rc_phase, rr_phase_raw = self.get_relation(r_idx)
        rc_phase = rc_phase / (self.embedding_range / pi)
        rr_phase = rr_phase_raw / (self.embedding_range / pi)

        rc_re = torch.cos(rc_phase)
        rc_im = torch.sin(rc_phase)
        # simplistic bound mapping for interval RotatE
        rr_bound = torch.abs(rr_phase)
        
        hc, hr = self.entity_emb(h_idx)
        tc, tr = self.entity_emb(t_idx)

        hc_re, hc_im = torch.chunk(hc, 2, dim=-1)
        hr_re, hr_im = torch.chunk(hr, 2, dim=-1)
        
        tc_re, tc_im = torch.chunk(tc, 2, dim=-1)
        tr_re, tr_im = torch.chunk(tr, 2, dim=-1)

        # Rotation application center
        pred_c_re = hc_re * rc_re - hc_im * rc_im
        pred_c_im = hc_re * rc_im + hc_im * rc_re
        
        # Simple interval propagation bound (rough upper bound via triangle ineq)
        pred_r_re = hr_re * torch.abs(rc_re) + hr_im * torch.abs(rc_im) + rr_bound * (torch.abs(hc_re) + torch.abs(hc_im))
        pred_r_im = hr_re * torch.abs(rc_im) + hr_im * torch.abs(rc_re) + rr_bound * (torch.abs(hc_re) + torch.abs(hc_im))

        diff_c_re = pred_c_re - tc_re
        diff_c_im = pred_c_im - tc_im
        
        diff_r_re = pred_r_re + tr_re
        diff_r_im = pred_r_im + tr_im

        dist_c = torch.norm(torch.stack([diff_c_re, diff_c_im], dim=0), dim=0).sum(dim=-1)
        dist_r = torch.norm(torch.stack([diff_r_re, diff_r_im], dim=0), dim=0).sum(dim=-1)

        return dist_r - dist_c # Max radius bound - distance

    def forward(self, pos_triplets: torch.Tensor, neg_triplets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pi = math.pi
        
        # Relations only have phase, so we project them to phase angles
        rc_phase, rr_phase_raw = self.get_relation(pos_triplets[:, 1])
        rc_phase = rc_phase / (self.embedding_range / pi)
        rr_phase = rr_phase_raw / (self.embedding_range / pi)

        rc_re = torch.cos(rc_phase)
        rc_im = torch.sin(rc_phase)
        # simplistic bound mapping for interval RotatE
        rr_bound = torch.abs(rr_phase)

        pos_h_idx = pos_triplets[:, 0]
        pos_t_idx = pos_triplets[:, 2]
        neg_h_idx = neg_triplets[:, :, 0]
        neg_t_idx = neg_triplets[:, :, 2]

        def compute_score(h_idx, t_idx, rc_re, rc_im, rr_bound):
            hc, hr = self.entity_emb(h_idx)
            tc, tr = self.entity_emb(t_idx)

            hc_re, hc_im = torch.chunk(hc, 2, dim=-1)
            hr_re, hr_im = torch.chunk(hr, 2, dim=-1)
            
            tc_re, tc_im = torch.chunk(tc, 2, dim=-1)
            tr_re, tr_im = torch.chunk(tr, 2, dim=-1)

            # Rotation application center
            pred_c_re = hc_re * rc_re - hc_im * rc_im
            pred_c_im = hc_re * rc_im + hc_im * rc_re
            
            # Simple interval propagation bound (rough upper bound via triangle ineq)
            pred_r_re = hr_re * torch.abs(rc_re) + hr_im * torch.abs(rc_im) + rr_bound * (torch.abs(hc_re) + torch.abs(hc_im))
            pred_r_im = hr_re * torch.abs(rc_im) + hr_im * torch.abs(rc_re) + rr_bound * (torch.abs(hc_re) + torch.abs(hc_im))

            diff_c_re = pred_c_re - tc_re
            diff_c_im = pred_c_im - tc_im
            
            diff_r_re = pred_r_re + tr_re
            diff_r_im = pred_r_im + tr_im

            dist_c = torch.norm(torch.stack([diff_c_re, diff_c_im], dim=0), dim=0).sum(dim=-1)
            dist_r = torch.norm(torch.stack([diff_r_re, diff_r_im], dim=0), dim=0).sum(dim=-1)

            return dist_r - dist_c # Max radius bound - distance

        pos_scores = compute_score(pos_h_idx, pos_t_idx, rc_re, rc_im, rr_bound)
        
        # Negative scoring: needs expanded rc and rr
        rc_re_neg = rc_re.unsqueeze(1)
        rc_im_neg = rc_im.unsqueeze(1)
        rr_bound_neg = rr_bound.unsqueeze(1)
        
        neg_scores = compute_score(neg_h_idx, neg_t_idx, rc_re_neg, rc_im_neg, rr_bound_neg)

        return pos_scores, neg_scores
