from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from core.evaluation import evaluate_model
from core.model_utils import load_model_from_checkpoint
from core.metrics import format_metrics_table
from data.dataset import load_dataset
from utils.runtime import (
    configure_logging,
    select_device,
    set_seed,
    setup_tensorboard,
    _RUN_NAME,
)

LOGGER = configure_logging()

def run_test(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    LOGGER.info("[ACTION] Loading dataset from %s", cfg.dataset.path)
    dataset = load_dataset(cfg.dataset.path)
    all_triples = torch.cat([dataset.train, dataset.valid, dataset.test], dim=0)

    model_type = cfg.model.name
    ckpt_name = cfg.get("checkpoint", f"{model_type}_{cfg.dataset.name}.pt")
    ckpt_path = Path(cfg.checkpoint_dir) / ckpt_name

    model, checkpoint_data, resolved_model_type = load_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        device=device,
        default_dim=cfg.model.dim,
        default_margin=cfg.model.margin,
        forced_model_type=model_type,
        hidden_layers=cfg.model.get("hidden_layers", []),
    )

    split = cfg.evaluation.split
    split_tensor = dataset.valid if split == "valid" else dataset.test

    writer = setup_tensorboard(cfg.tensorboard_dir, _RUN_NAME)

    try:
        writer.add_text("hyperparameters", str(cfg))

        LOGGER.info(
            "[ACTION] Starting evaluation on %s split...", split
        )
        metrics = None
        metrics = evaluate_model(
            model=model,
            split_tensor=split_tensor,
            all_triples=all_triples,
            num_entities=dataset.num_entities,
            device=device,
            num_negatives=cfg.evaluation.num_negatives,
            batch_size=cfg.evaluation.get("batch_size", cfg.training.batch_size),
            entity_chunk_size=cfg.evaluation.entity_chunk_size,
        )

        for key, value in metrics.items():
            writer.add_scalar(f"Test_Metrics/{key}", value, 0)

    finally:
        writer.close()

        if metrics is not None:
            # Format and display results nicely
            metrics_table = format_metrics_table(
                metrics,
                title=f"Evaluation Results ({split.upper()} split)",
            )
            LOGGER.info("\n%s", metrics_table)
            LOGGER.info(
                "[ACTION] Evaluation complete (%s, %s): MRR=%.6f, Hits@1=%.4f, Hits@3=%.4f, Hits@10=%.4f",
                split,
                resolved_model_type,
                metrics["mrr"],
                metrics["hits_at_1"],
                metrics["hits_at_3"],
                metrics["hits_at_10"],
            )
