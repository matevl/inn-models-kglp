"""Evaluation utilities for link prediction."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_approx_ranking(
    model: nn.Module,
    split_triples: torch.Tensor,
    all_triples: torch.Tensor,
    num_entities: int,
    device: torch.device,
    num_negatives: int,
    batch_size: int,
    entity_chunk_size: int,  # Added entity_chunk_size to control sub-batch limits
) -> Dict[str, float]:
    """Evaluate using approximate ranking with sampled negatives.

    Args:
        model (nn.Module): The link prediction model to evaluate.
        split_triples (torch.Tensor): A tensor of shape (num_triples, 3) representing the test or validation triples.
        all_triples (torch.Tensor): A tensor of shape (num_all_triples, 3) representing all known triples for filtering.
        num_entities (int): Total number of entities in the knowledge graph.
        device (torch.device): The device on which to perform evaluation computations.
        num_negatives (int): Number of negative samples to generate per positive triple for approximate ranking.
        batch_size (int): The number of triples to process per batch.
        entity_chunk_size (int): The chunk size for evaluating negative entities to manage memory usage.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics: 'mrr', 'hits_at_1', 'hits_at_3', and 'hits_at_10'.
    """
    model.eval()
    loader = DataLoader(
        split_triples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build lookup for filtering
    filter_hr = defaultdict(list)
    filter_rt = defaultdict(list)
    for h, r, t in all_triples.tolist():
        filter_hr[(h, r)].append(t)
        filter_rt[(r, t)].append(h)
    filter_hr = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in filter_hr.items()
    }
    filter_rt = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in filter_rt.items()
    }

    rr_total = 0.0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    n_total = 0

    for batch in loader:
        batch = batch.to(device)
        bsz = batch.size(0)

        true_h = batch[:, 0]
        true_r = batch[:, 1]
        true_t = batch[:, 2]

        true_scores = model.inn_score(true_h, true_r, true_t)

        # Right Rank (negate tails)
        neg_tails = torch.randint(
            low=0,
            high=num_entities,
            size=(bsz, num_negatives),
            device=device,
        )

        for i in range(bsz):
            h, r = true_h[i].item(), true_r[i].item()
            if (h, r) in filter_hr:
                known_tails = filter_hr[(h, r)]
                while True:
                    collisions = torch.isin(neg_tails[i], known_tails)
                    if not collisions.any():
                        break
                    neg_tails[i][collisions] = torch.randint(
                        low=0,
                        high=num_entities,
                        size=(collisions.sum().item(),),
                        device=device,
                    )

        # Left Rank (negate heads)
        neg_heads = torch.randint(
            low=0,
            high=num_entities,
            size=(bsz, num_negatives),
            device=device,
        )

        for i in range(bsz):
            r, t = true_r[i].item(), true_t[i].item()
            if (r, t) in filter_rt:
                known_heads = filter_rt[(r, t)]
                while True:
                    collisions = torch.isin(neg_heads[i], known_heads)
                    if not collisions.any():
                        break
                    neg_heads[i][collisions] = torch.randint(
                        low=0,
                        high=num_entities,
                        size=(collisions.sum().item(),),
                        device=device,
                    )

        max_chunk_elems = max(entity_chunk_size, 64 * num_negatives)

        # Right Rank evaluation
        rep_h_rt = true_h.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        rep_r_rt = true_r.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        flat_t_rt = neg_tails.reshape(-1)

        neg_scores_list_rt = []
        for start_idx in range(0, rep_h_rt.size(0), max_chunk_elems):
            end_idx = start_idx + max_chunk_elems
            neg_scores_list_rt.append(
                model.inn_score(
                    rep_h_rt[start_idx:end_idx],
                    rep_r_rt[start_idx:end_idx],
                    flat_t_rt[start_idx:end_idx],
                )
            )

        neg_scores_rt = torch.cat(neg_scores_list_rt, dim=0).reshape(bsz, num_negatives)
        ranks_rt = 1 + torch.sum(neg_scores_rt > true_scores.unsqueeze(1), dim=1)

        # Left Rank evaluation
        flat_h_lf = neg_heads.reshape(-1)
        rep_r_lf = true_r.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        rep_t_lf = true_t.unsqueeze(1).expand(-1, num_negatives).reshape(-1)

        neg_scores_list_lf = []
        for start_idx in range(0, flat_h_lf.size(0), max_chunk_elems):
            end_idx = start_idx + max_chunk_elems
            neg_scores_list_lf.append(
                model.inn_score(
                    flat_h_lf[start_idx:end_idx],
                    rep_r_lf[start_idx:end_idx],
                    rep_t_lf[start_idx:end_idx],
                )
            )

        neg_scores_lf = torch.cat(neg_scores_list_lf, dim=0).reshape(bsz, num_negatives)
        ranks_lf = 1 + torch.sum(neg_scores_lf > true_scores.unsqueeze(1), dim=1)

        for ranks in (ranks_rt, ranks_lf):
            rr_total += torch.sum(1.0 / ranks.float()).item()
            hits1 += torch.sum(ranks <= 1).item()
            hits3 += torch.sum(ranks <= 3).item()
            hits10 += torch.sum(ranks <= 10).item()
            n_total += bsz

    if n_total == 0:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_10": 0.0}

    return {
        "mrr": rr_total / n_total,
        "hits_at_1": hits1 / n_total,
        "hits_at_3": hits3 / n_total,
        "hits_at_10": hits10 / n_total,
    }


@torch.no_grad()
def evaluate_exact_ranking_all_entities(
    model: nn.Module,
    split_triples: torch.Tensor,
    all_triples: torch.Tensor,
    num_entities: int,
    device: torch.device,
    batch_size: int,
    entity_chunk_size: int,
) -> Dict[str, float]:
    """Evaluate using exact ranking against all entities in the dataset.

    Args:
        model (nn.Module): The link prediction model to evaluate.
        split_triples (torch.Tensor): A tensor of shape (num_triples, 3) representing the test or validation triples.
        all_triples (torch.Tensor): A tensor of shape (num_all_triples, 3) representing all known triples for filtering.
        num_entities (int): Total number of entities in the knowledge graph.
        device (torch.device): The device on which to perform evaluation computations.
        batch_size (int): The number of triples to process per batch.
        entity_chunk_size (int): The chunk size for evaluating entities to manage memory usage.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics: 'mrr', 'hits_at_1', 'hits_at_3', and 'hits_at_10'.
    """
    model.eval()
    loader = DataLoader(
        split_triples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    from collections import defaultdict

    filter_hr = defaultdict(list)
    filter_rt = defaultdict(list)
    for h, r, t in all_triples.tolist():
        filter_hr[(h, r)].append(t)
        filter_rt[(r, t)].append(h)
    filter_hr = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in filter_hr.items()
    }
    filter_rt = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in filter_rt.items()
    }

    rr_total = 0.0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    n_total = 0

    for batch in loader:
        batch = batch.to(device)
        bsz = batch.size(0)
        true_h = batch[:, 0]
        true_r = batch[:, 1]
        true_t = batch[:, 2]

        true_scores = model.inn_score(true_h, true_r, true_t)

        rank_counts_rt = torch.zeros(bsz, dtype=torch.long, device=device)
        rank_counts_lf = torch.zeros(bsz, dtype=torch.long, device=device)

        for start in range(0, num_entities, entity_chunk_size):
            end = min(start + entity_chunk_size, num_entities)
            chunk_size = end - start

            chunk_ents = torch.arange(start, end, device=device)

            # Right Rank chunk
            rep_h_rt = true_h.unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            rep_r_rt = true_r.unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            rep_t_rt = chunk_ents.unsqueeze(0).expand(bsz, -1).reshape(-1)

            chunk_scores_rt = model.inn_score(rep_h_rt, rep_r_rt, rep_t_rt).reshape(
                bsz, chunk_size
            )

            # Left Rank chunk
            rep_h_lf = chunk_ents.unsqueeze(0).expand(bsz, -1).reshape(-1)
            rep_r_lf = true_r.unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            rep_t_lf = true_t.unsqueeze(1).expand(-1, chunk_size).reshape(-1)

            chunk_scores_lf = model.inn_score(rep_h_lf, rep_r_lf, rep_t_lf).reshape(
                bsz, chunk_size
            )

            for i in range(bsz):
                h, r, t = true_h[i].item(), true_r[i].item(), true_t[i].item()

                # Filter Right Rank
                if (h, r) in filter_hr:
                    known_tails = filter_hr[(h, r)]
                    known_tails_to_mask = known_tails[known_tails != t]
                    known_in_chunk = (
                        known_tails_to_mask[
                            (known_tails_to_mask >= start) & (known_tails_to_mask < end)
                        ]
                        - start
                    )
                    if len(known_in_chunk) > 0:
                        chunk_scores_rt[i, known_in_chunk] = -1e9

                # Filter Left Rank
                if (r, t) in filter_rt:
                    known_heads = filter_rt[(r, t)]
                    known_heads_to_mask = known_heads[known_heads != h]
                    known_in_chunk = (
                        known_heads_to_mask[
                            (known_heads_to_mask >= start) & (known_heads_to_mask < end)
                        ]
                        - start
                    )
                    if len(known_in_chunk) > 0:
                        chunk_scores_lf[i, known_in_chunk] = -1e9

            rank_counts_rt += torch.sum(
                chunk_scores_rt > true_scores.unsqueeze(1), dim=1
            )
            rank_counts_lf += torch.sum(
                chunk_scores_lf > true_scores.unsqueeze(1), dim=1
            )

        ranks_rt = 1 + rank_counts_rt
        ranks_lf = 1 + rank_counts_lf

        for ranks in (ranks_rt, ranks_lf):
            rr_total += torch.sum(1.0 / ranks.float()).item()
            hits1 += torch.sum(ranks <= 1).item()
            hits3 += torch.sum(ranks <= 3).item()
            hits10 += torch.sum(ranks <= 10).item()
            n_total += bsz

    if n_total == 0:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_10": 0.0}

    return {
        "mrr": rr_total / n_total,
        "hits_at_1": hits1 / n_total,
        "hits_at_3": hits3 / n_total,
        "hits_at_10": hits10 / n_total,
    }


def evaluate_model(
    model: nn.Module,
    split_tensor: torch.Tensor,
    all_triples: torch.Tensor,
    num_entities: int,
    device: torch.device,
    num_negatives: int,
    batch_size: int,
    entity_chunk_size: int,
) -> Dict[str, float]:
    """Route to exact or approximate evaluation based on num_negatives.

    Args:
        model (nn.Module): The link prediction model to evaluate.
        split_tensor (torch.Tensor): A tensor of shape (num_triples, 3) representing the evaluation triples.
        all_triples (torch.Tensor): A tensor of shape (num_all_triples, 3) representing all known triples for filtering.
        num_entities (int): Total number of entities in the knowledge graph.
        device (torch.device): The device on which to perform evaluation computations.
        num_negatives (int): Number of negative samples per positive triple. If < 0, exact ranking is used.
        batch_size (int): The number of triples to process per batch.
        entity_chunk_size (int): The chunk size for evaluating entities to manage memory usage.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics: 'mrr', 'hits_at_1', 'hits_at_3', and 'hits_at_10'.
    """
    if num_negatives < 0 or num_negatives >= num_entities - 1:
        if entity_chunk_size <= 0:
            raise ValueError("--entity-chunk-size must be > 0")
        return evaluate_exact_ranking_all_entities(
            model=model,
            split_triples=split_tensor,
            all_triples=all_triples,
            num_entities=num_entities,
            device=device,
            batch_size=batch_size,
            entity_chunk_size=entity_chunk_size,
        )

    return evaluate_approx_ranking(
        model=model,
        split_triples=split_tensor,
        all_triples=all_triples,
        num_entities=num_entities,
        device=device,
        num_negatives=num_negatives,
        batch_size=batch_size,
        entity_chunk_size=entity_chunk_size,
    )
