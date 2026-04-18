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
) -> Dict[str, float]:
    """Evaluate using approximate ranking with sampled negatives."""
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
    for h, r, t in all_triples.tolist():
        filter_hr[(h, r)].append(t)
    filter_hr = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in filter_hr.items()
    }

    rr_total = 0.0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    n_total = 0

    for batch in loader:
        batch = batch.to(device)
        bsz = batch.size(0)

        true_scores = model.inn_score(batch[:, 0], batch[:, 1], batch[:, 2])

        neg_tails = torch.randint(
            low=0,
            high=num_entities,
            size=(bsz, num_negatives),
            device=device,
        )

        # Replace negatives that are actually true triples with random entities until valid
        for i in range(bsz):
            h, r, t = batch[i, 0].item(), batch[i, 1].item(), batch[i, 2].item()
            if (h, r) in filter_hr:
                known_tails = filter_hr[(h, r)]

                # Identify collisions
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

        rep_h = batch[:, 0].unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        rep_r = batch[:, 1].unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        flat_t = neg_tails.reshape(-1)

        neg_scores = model.inn_score(rep_h, rep_r, flat_t).reshape(bsz, num_negatives)
        ranks = 1 + torch.sum(neg_scores > true_scores.unsqueeze(1), dim=1)

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
    """Evaluate using exact ranking against all entities."""
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
    for h, r, t in all_triples.tolist():
        filter_hr[(h, r)].append(t)
    filter_hr = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in filter_hr.items()
    }

    rr_total = 0.0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    n_total = 0

    for batch in loader:
        batch = batch.to(device)
        bsz = batch.size(0)
        true_tails = batch[:, 2]
        true_scores = model.inn_score(batch[:, 0], batch[:, 1], true_tails)

        rank_counts = torch.zeros(bsz, dtype=torch.long, device=device)
        for start in range(0, num_entities, entity_chunk_size):
            end = min(start + entity_chunk_size, num_entities)
            chunk_size = end - start

            chunk_tails = torch.arange(start, end, device=device)
            rep_h = batch[:, 0].unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            rep_r = batch[:, 1].unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            rep_t = chunk_tails.unsqueeze(0).expand(bsz, -1).reshape(-1)

            chunk_scores = model.inn_score(rep_h, rep_r, rep_t).reshape(bsz, chunk_size)

            # Filter known positive tails from counting towards rank
            # Mask all known positive tails EXCEPT the true tail of the current query
            for i in range(bsz):
                h, r = batch[i, 0].item(), batch[i, 1].item()
                true_t = true_tails[i].item()
                if (h, r) in filter_hr:
                    known_tails = filter_hr[(h, r)]
                    # Exclude the current true tail from known tails to be masked
                    known_tails_to_mask = known_tails[known_tails != true_t]
                    known_in_chunk = (
                        known_tails_to_mask[
                            (known_tails_to_mask >= start) & (known_tails_to_mask < end)
                        ]
                        - start
                    )
                    if len(known_in_chunk) > 0:
                        chunk_scores[i, known_in_chunk] = -1e9

            rank_counts += torch.sum(chunk_scores > true_scores.unsqueeze(1), dim=1)

        ranks = 1 + rank_counts

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
    """Route to exact or approximate evaluation based on num_negatives."""
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
    )
