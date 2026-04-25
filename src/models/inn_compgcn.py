import torch
import torch.nn as nn
import torch.nn.functional as F

from .inn_ours_mlp import IntervalEntityEmbedding
from core.intervals import Interval, interval_relu


class CompGCNIntervalLayer(nn.Module):
    def __init__(self, in_dim, out_dim, init_rho: float = -5.0):
        super().__init__()
        self.W_in = nn.Linear(in_dim, out_dim, bias=False)
        self.W_out = nn.Linear(in_dim, out_dim, bias=False)
        self.W_loop = nn.Linear(in_dim, out_dim, bias=False)
        
        self.W_rel = nn.Linear(in_dim, out_dim, bias=False)
        
        self.loop_rel_c = nn.Parameter(torch.zeros(1, in_dim))
        self.loop_rel_r = nn.Parameter(torch.full((1, in_dim), init_rho))

    def forward(self, H: Interval, rel_c: torch.Tensor, rel_r: torch.Tensor, in_edges, out_edges, loop_edges):
        num_ent = H.c.shape[0]
        
        in_row, in_col, in_type, in_norm = in_edges
        out_row, out_col, out_type, out_norm = out_edges
        loop_row, loop_col = loop_edges
        
        def aggregate(row, col, edge_type, r_c, r_r, W, mode, edge_weight):
            x_j_c = H.c[col]
            x_j_r = H.r[col]
            
            if mode == "in":
                msg_c = x_j_c + r_c[edge_type]
                msg_r = x_j_r + r_r[edge_type]
            elif mode == "out":
                msg_c = x_j_c - r_c[edge_type]
                msg_r = x_j_r + r_r[edge_type]
            else:
                msg_c = x_j_c + r_c.expand(len(col), -1)
                msg_r = x_j_r + r_r.expand(len(col), -1)
                
            out_c = msg_c @ W.weight.t()
            out_r = msg_r @ W.weight.abs().t()
            
            if edge_weight is not None:
                out_c = out_c * edge_weight.unsqueeze(-1)
                out_r = out_r * edge_weight.unsqueeze(-1)
                
            dim = out_c.shape[1]
            res_c = torch.zeros(num_ent, dim, device=out_c.device, dtype=out_c.dtype)
            res_r = torch.zeros(num_ent, dim, device=out_r.device, dtype=out_r.dtype)
            
            index = row.unsqueeze(-1).expand(-1, dim)
            res_c.scatter_add_(0, index, out_c)
            res_r.scatter_add_(0, index, out_r)
            
            return res_c, res_r

        c_in, r_in = aggregate(in_row, in_col, in_type, rel_c, rel_r, self.W_in, "in", in_norm)
        c_out, r_out = aggregate(out_row, out_col, out_type, rel_c, rel_r, self.W_out, "out", out_norm)
        
        soft_loop_r = F.softplus(self.loop_rel_r)
        c_loop, r_loop = aggregate(loop_row, loop_col, None, self.loop_rel_c, soft_loop_r, self.W_loop, "loop", None)
        
        # Average aggregations across edge directions (in, out, loop)
        c_agg = (c_in + c_out + c_loop) / 3.0
        r_agg = (r_in + r_out + r_loop) / 3.0
        
        # Propagate relation embeddings
        new_rel_c = self.W_rel(rel_c)
        new_rel_r = rel_r @ self.W_rel.weight.abs().t()
        
        Hn = Interval(c_agg, r_agg)
        return interval_relu(Hn), new_rel_c, new_rel_r


class INNCompGCNLinkPredictor(nn.Module):
    """Interval-based implementation of CompGCN."""

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
        self.layer = CompGCNIntervalLayer(dim, dim, init_rho=init_rho)
        self.gamma_margin = gamma_margin

        nn.init.uniform_(self.rel_center.weight, -0.1, 0.1)
        nn.init.constant_(self.rel_rho.weight, init_rho)
        
        self.register_buffer("in_row", None, persistent=False)
        self.register_buffer("in_col", None, persistent=False)
        self.register_buffer("in_type", None, persistent=False)
        self.register_buffer("in_norm", None, persistent=False)
        
        self.register_buffer("out_row", None, persistent=False)
        self.register_buffer("out_col", None, persistent=False)
        self.register_buffer("out_type", None, persistent=False)
        self.register_buffer("out_norm", None, persistent=False)
        
        self.register_buffer("loop_row", None, persistent=False)
        self.register_buffer("loop_col", None, persistent=False)

    def build_graph(self, train_triples: torch.Tensor) -> None:
        """Construct the graph components for message passing.
        Note: If data leakage occurs (accuracy 99%), edge dropout should be applied here.
        """
        num_ent = self.entity_emb.center.num_embeddings
        device = self.entity_emb.center.weight.device

        # IN edges
        self.in_row = train_triples[:, 2].to(device)
        self.in_col = train_triples[:, 0].to(device)
        self.in_type = train_triples[:, 1].to(device)

        # OUT edges
        self.out_row = train_triples[:, 0].to(device)
        self.out_col = train_triples[:, 2].to(device)
        self.out_type = train_triples[:, 1].to(device)

        # LOOP edges
        self.loop_row = torch.arange(num_ent, device=device)
        self.loop_col = torch.arange(num_ent, device=device)

        def compute_norm(row, col):
            deg = torch.bincount(row, minlength=num_ent).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            return deg_inv_sqrt[row] * deg_inv_sqrt[col]

        self.in_norm = compute_norm(self.in_row, self.in_col)
        self.out_norm = compute_norm(self.out_row, self.out_col)

    def get_relation(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_r = self.rel_center(idx)
        r_r = F.softplus(self.rel_rho(idx))
        return c_r, r_r

    def compute_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_ent = self.entity_emb.center.num_embeddings
        device = self.entity_emb.center.weight.device
        
        all_entity_ids = torch.arange(num_ent, device=device)
        u_c, u_r = self.entity_emb(all_entity_ids)
        
        all_rel_ids = torch.arange(self.rel_center.num_embeddings, device=device)
        rel_c, rel_r = self.get_relation(all_rel_ids)

        if self.in_row is not None:
            H = Interval(u_c, u_r)
            in_edges = (self.in_row, self.in_col, self.in_type, self.in_norm)
            out_edges = (self.out_row, self.out_col, self.out_type, self.out_norm)
            loop_edges = (self.loop_row, self.loop_col)
            
            Hn, rel_c, rel_r = self.layer(H, rel_c, rel_r, in_edges, out_edges, loop_edges)
            return Hn.c, Hn.r, rel_c, rel_r
            
        return u_c, u_r, rel_c, rel_r

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
        u_c, u_r, rel_c, rel_r = self.compute_all_embeddings()
        
        hc, hr = u_c[h_idx], u_r[h_idx]
        tc, tr = u_c[t_idx], u_r[t_idx]
        rc, rr = rel_c[r_idx], rel_r[r_idx]

        pred_c = hc + rc
        pred_r = hr + rr

        distance = torch.norm(pred_c - tc, p=1, dim=-1)
        max_radius_sum = torch.norm(pred_r + tr, p=1, dim=-1)
        return max_radius_sum - distance

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
        u_c, u_r, rel_c, rel_r = self.compute_all_embeddings()

        pos_h_idx = pos_triplets[:, 0]
        pos_t_idx = pos_triplets[:, 2]
        pos_r_idx = pos_triplets[:, 1]
        
        neg_h_idx = neg_triplets[:, :, 0]
        neg_t_idx = neg_triplets[:, :, 2]

        # Score computation (simplified TransE)
        hc, hr = u_c[pos_h_idx], u_r[pos_h_idx]
        tc, tr = u_c[pos_t_idx], u_r[pos_t_idx]
        rc, rr = rel_c[pos_r_idx], rel_r[pos_r_idx]
        
        pred_c = hc + rc
        pred_r = hr + rr
        
        distance = torch.norm(pred_c - tc, p=1, dim=-1)
        max_radius_sum = torch.norm(pred_r + tr, p=1, dim=-1)
        pos_scores = max_radius_sum - distance

        hc_neg, hr_neg = u_c[neg_h_idx], u_r[neg_h_idx]
        tc_neg, tr_neg = u_c[neg_t_idx], u_r[neg_t_idx]
        rc_neg, rr_neg = rc.unsqueeze(1), rr.unsqueeze(1)
        
        pred_c_neg = hc_neg + rc_neg
        pred_r_neg = hr_neg + rr_neg
        
        distance_neg = torch.norm(pred_c_neg - tc_neg, p=1, dim=-1)
        max_radius_sum_neg = torch.norm(pred_r_neg + tr_neg, p=1, dim=-1)
        neg_scores = max_radius_sum_neg - distance_neg

        return pos_scores, neg_scores
