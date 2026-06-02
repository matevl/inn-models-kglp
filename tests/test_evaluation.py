from __future__ import annotations

import torch

from src.core.evaluation import _compute_cached_embeddings, _score_triples
from src.models import build_link_predictor


def test_lightgcn_cached_embeddings_score_matches_model():
    model = build_link_predictor(
        model_type="inn_lightgcn",
        num_entities=6,
        num_relations=3,
        dim=8,
        gamma_margin=1.0,
        init_rho=-5.0,
        hidden_layers=None,
        edge_dropout_p=0.0,
    )
    model.eval()

    triples = torch.tensor([[0, 1, 2], [3, 2, 4]], dtype=torch.long)

    cached_embeddings = _compute_cached_embeddings(model)
    scores_from_cache = _score_triples(
        model,
        cached_embeddings,
        triples[:, 0],
        triples[:, 1],
        triples[:, 2],
    )
    scores_from_model = model.inn_score(triples[:, 0], triples[:, 1], triples[:, 2])

    assert cached_embeddings is not None
    assert len(cached_embeddings) == 2
    assert torch.allclose(scores_from_cache, scores_from_model)
