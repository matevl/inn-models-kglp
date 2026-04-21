import torch
import torch.nn as nn
import torch.nn.functional as F

from .inn_ours_mlp import IntervalEntityEmbedding
from .inn_lightgcn import TInterval


class CompGCNIntervalLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_in = nn.Linear(in_dim, out_dim, bias=False)
        self.W_out = nn.Linear(in_dim, out_dim, bias=False)
        self.W_loop = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, A_in, A_out, A_loop, H: TInterval):
        # In-edges
        c_in = self.W_in(H.c)
        r_in = H.r @ self.W_in.weight.abs().t()

        # Out-edges
        c_out = self.W_out(H.c)
        r_out = H.r @ self.W_out.weight.abs().t()

        # Loop-edges
        c_loop = self.W_loop(H.c)
        r_loop = H.r @ self.W_loop.weight.abs().t()

        # Aggregation
        # PyTorch does not support sparse mm in FP16, so we must disable autocast and run in FP32
        device_type = c_in.device.type
        with torch.autocast(device_type=device_type if device_type != 'mps' else 'cpu', enabled=False):
            A_in_f32, c_in_f32 = A_in.to(torch.float32), c_in.to(torch.float32)
            A_out_f32, c_out_f32 = A_out.to(torch.float32), c_out.to(torch.float32)
            A_loop_f32, c_loop_f32 = A_loop.to(torch.float32), c_loop.to(torch.float32)
            r_in_f32, r_out_f32, r_loop_f32 = r_in.to(torch.float32), r_out.to(torch.float32), r_loop.to(torch.float32)
            
            c_agg_f32 = A_in_f32 @ c_in_f32 + A_out_f32 @ c_out_f32 + A_loop_f32 @ c_loop_f32
            r_agg_f32 = A_in_f32.abs() @ r_in_f32 + A_out_f32.abs() @ r_out_f32 + A_loop_f32.abs() @ r_loop_f32
            
        c_agg = c_agg_f32.to(c_in.dtype)
        r_agg = r_agg_f32.to(r_in.dtype)

        # Activation (ReLU)
        Hn = TInterval(c_agg, r_agg)
        lo, hi = Hn.lu()
        return TInterval.from_lu(F.relu(lo), F.relu(hi))


class INNCompGCNLinkPredictor(nn.Module):
    """Interval-based implementation of CompGCN."""

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
        self.layer = CompGCNIntervalLayer(dim, dim)
        self.margin = margin

        nn.init.uniform_(self.rel_center.weight, -0.1, 0.1)
        nn.init.constant_(self.rel_rho.weight, init_rho)
        self.register_buffer("A_in", None, persistent=False)
        self.register_buffer("A_out", None, persistent=False)
        self.register_buffer("A_loop", None, persistent=False)

    def build_graph(self, train_triples: torch.Tensor) -> None:
        num_ent = self.entity_emb.center.num_embeddings
        device = self.entity_emb.center.weight.device

        # In edges
        edges_in = train_triples[:, [0, 2]].t().to(device)
        edges_out = train_triples[:, [2, 0]].t().to(device)
        self_loops = torch.arange(num_ent, device=device).unsqueeze(0).repeat(2, 1)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
            
            def make_sparse(edges):
                row, col = edges
                deg = torch.bincount(row, minlength=num_ent).float()
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
                edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                return torch.sparse_coo_tensor(edges, edge_weight, (num_ent, num_ent)).to(device)

            self.A_in = make_sparse(edges_in)
            self.A_out = make_sparse(edges_out)

            # Self loops without normalization (just identity)
            loop_weight = torch.ones(num_ent, device=device)
            self.A_loop = torch.sparse_coo_tensor(
                self_loops, loop_weight, (num_ent, num_ent)
            ).to(device)

    def get_relation(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_r = self.rel_center(idx)
        r_r = F.softplus(self.rel_rho(idx))
        return c_r, r_r

    def compute_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_ent = self.entity_emb.center.num_embeddings
        device = self.entity_emb.center.weight.device
        all_entity_ids = torch.arange(num_ent, device=device)
        u_c, u_r = self.entity_emb(all_entity_ids)

        if self.A_in is not None and self.A_out is not None and self.A_loop is not None:
            H = TInterval(u_c, u_r)
            H = self.layer(self.A_in, self.A_out, self.A_loop, H)
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
        self, pos_triplets: torch.Tensor, neg_triplets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rc, rr = self.get_relation(pos_triplets[:, 1])
        u_c, u_r = self.compute_all_embeddings()

        pos_h_idx = pos_triplets[:, 0]
        pos_t_idx = pos_triplets[:, 2]
        neg_h_idx = neg_triplets[:, :, 0]
        neg_t_idx = neg_triplets[:, :, 2]

        # Score computation (simplified TransE)
        hc, hr = u_c[pos_h_idx], u_r[pos_h_idx]
        tc, tr = u_c[pos_t_idx], u_r[pos_t_idx]
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
