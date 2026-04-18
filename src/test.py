from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.evaluation import evaluate_model
from core.model_utils import load_model_from_checkpoint
from core.metrics import format_metrics_table
from data.dataset import load_dataset
from utils.runtime import (
    configure_logging,
    select_device,
    set_seed,
    setup_tensorboard,
)

LOGGER = configure_logging()


def run_test(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)

    LOGGER.info("[ACTION] Loading dataset from %s", args.dataset)
    dataset = load_dataset(args.dataset)
    all_triples = torch.cat([dataset.train, dataset.valid, dataset.test], dim=0)

    model, checkpoint_data, resolved_model_type = load_model_from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        default_dim=args.dim,
        default_margin=args.margin,
        forced_model_type=args.model_type,
        hidden_dim_coef=args.hidden_dim_coef,
    )

    split_tensor = dataset.valid if args.split == "valid" else dataset.test

    # Implement bidirectional evaluation: for each triple (h, r, t), add both
    # - direct query: (h, r, ?) with target t
    # - inverse query: (t, r + num_orig_relations, ?) with target h
    num_orig_relations = dataset.num_relations // 2
    inverse_queries = torch.zeros((split_tensor.size(0), 3), dtype=torch.long)
    inverse_queries[:, 0] = split_tensor[:, 2]  # head = original tail
    inverse_queries[:, 1] = (
        split_tensor[:, 1] + num_orig_relations
    )  # rel = r + num_orig_relations
    inverse_queries[:, 2] = split_tensor[:, 0]  # tail = original head

    # Concatenate direct and inverse queries
    eval_triples = torch.cat([split_tensor, inverse_queries], dim=0)

    writer = setup_tensorboard(args.tensorboard_dir)

    try:
        writer.add_text("hyperparameters", str(vars(args)))

        LOGGER.info(
            "[ACTION] Starting bidirectional evaluation on %s split...", args.split
        )
        metrics = evaluate_model(
            model=model,
            split_tensor=eval_triples,
            all_triples=all_triples,
            num_entities=dataset.num_entities,
            device=device,
            num_negatives=args.num_negatives,
            batch_size=args.batch_size,
            entity_chunk_size=args.entity_chunk_size,
        )

        for key, value in metrics.items():
            writer.add_scalar(f"Test_Metrics/{key}", value, 0)

    finally:
        writer.close()

        # Format and display results nicely
        metrics_table = format_metrics_table(
            metrics,
            title=f"Bidirectional Evaluation Results ({args.split.upper()} split)",
        )
        LOGGER.info("\n%s", metrics_table)
        LOGGER.info(
            "[ACTION] Bidirectional evaluation complete (%s, %s): MRR=%.6f, Hits@1=%.4f, Hits@3=%.4f, Hits@10=%.4f",
            args.split,
            resolved_model_type,
            metrics["mrr"],
            metrics["hits_at_1"],
            metrics["hits_at_3"],
            metrics["hits_at_10"],
        )


def run_compare(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)

    LOGGER.info("[ACTION] Loading dataset from %s", args.dataset)
    dataset = load_dataset(args.dataset)
    split_tensor = dataset.valid if args.split == "valid" else dataset.test
    all_triples = torch.cat([dataset.train, dataset.valid, dataset.test], dim=0)

    first_model, _, first_type = load_model_from_checkpoint(
        checkpoint_path=Path(args.lhs_checkpoint),
        device=device,
        default_dim=args.dim,
        default_margin=args.margin,
        forced_model_type="auto",
        hidden_dim_coef=args.hidden_dim_coef,
    )
    first_metrics = evaluate_model(
        model=first_model,
        split_tensor=split_tensor,
        all_triples=all_triples,
        num_entities=dataset.num_entities,
        device=device,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        entity_chunk_size=args.entity_chunk_size,
    )
    del first_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    second_model, _, second_type = load_model_from_checkpoint(
        checkpoint_path=Path(args.rhs_checkpoint),
        device=device,
        default_dim=args.dim,
        default_margin=args.margin,
        forced_model_type="auto",
        hidden_dim_coef=args.hidden_dim_coef,
    )
    second_metrics = evaluate_model(
        model=second_model,
        split_tensor=split_tensor,
        all_triples=all_triples,
        num_entities=dataset.num_entities,
        device=device,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        entity_chunk_size=args.entity_chunk_size,
    )
    del second_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lhs_metrics = first_metrics
    rhs_metrics = second_metrics

    writer = setup_tensorboard(args.tensorboard_dir)
    try:
        writer.add_text("hyperparameters", str(vars(args)))

        for key, value in lhs_metrics.items():
            writer.add_scalar(f"Compare_{first_type}/{key}", value, 0)
        for key, value in rhs_metrics.items():
            writer.add_scalar(f"Compare_{second_type}/{key}", value, 0)
        for key in lhs_metrics:
            writer.add_scalar(
                f"Compare_Delta_{first_type}_minus_{second_type}/{key}",
                lhs_metrics[key] - rhs_metrics[key],
                0,
            )
    finally:
        writer.close()

    LOGGER.info("[ACTION] LHS (%s) metrics: %s", first_type, lhs_metrics)
    LOGGER.info("[ACTION] RHS (%s) metrics: %s", second_type, rhs_metrics)
    LOGGER.info(
        "[ACTION] Delta (%s - %s): %s",
        first_type,
        second_type,
        {k: lhs_metrics[k] - rhs_metrics[k] for k in lhs_metrics},
    )
