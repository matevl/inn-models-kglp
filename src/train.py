from __future__ import annotations
import logging
from pathlib import Path
from omegaconf import DictConfig

import torch

from core.training import train_epoch
from data.dataset import load_dataset
from models import build_link_predictor
from utils.runtime import (
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
    setup_tensorboard,
)
from utils.runtime import _RUN_NAME

LOGGER = logging.getLogger("inn-models-kglp")


def run_training(cfg: DictConfig, resume: bool) -> None:
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    LOGGER.info(
        "[ACTION] Loading dataset %s from %s", cfg.dataset.name, cfg.dataset.path
    )
    dataset = load_dataset(cfg.dataset.path)

    writer = setup_tensorboard(cfg.tensorboard_dir, _RUN_NAME)

    model_cfg = cfg.model
    model_type = cfg.model.name

    start_epoch = 1

    # Checkpoint logic
    ckpt_name = cfg.get("checkpoint", f"{model_type}_{cfg.dataset.name}.pt")
    ckpt_path = Path(cfg.checkpoint_dir) / ckpt_name

    if resume and ckpt_path.exists():
        LOGGER.info("[ACTION] Resuming from %s", ckpt_path)
        checkpoint_data = load_checkpoint(ckpt_path, device)
        # Assuming you load parameters correctly here...
        model = build_link_predictor(
            model_type=model_type,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            dim=model_cfg.dim,
            margin=model_cfg.margin,
            init_rho=model_cfg.init_rho,
            hidden_layers=model_cfg.get("hidden_layers", []),
        ).to(device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        # Set start_epoch...
    else:
        LOGGER.info(
            "[ACTION] Initializing new model %s",
            model_type,
        )
        model = build_link_predictor(
            model_type=model_type,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            dim=model_cfg.dim,
            margin=model_cfg.margin,
            init_rho=model_cfg.init_rho,
            hidden_layers=model_cfg.get("hidden_layers", []),
        ).to(device)

    if hasattr(model, "entity_encoder") and hasattr(model.entity_encoder, "entity_rho"):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    LOGGER.info(f"[ACTION] Starting training for {cfg.training.epochs} epochs...")
    model.train()

    epoch_loss_total = 0.0

    # Custom training loop to show logs dynamically
    for epoch in range(start_epoch, start_epoch + cfg.training.epochs):
        loss_dict = train_epoch(
            model=model,
            optimizer=optimizer,
            triples=dataset.train,
            batch_size=cfg.training.batch_size,
            num_negatives=cfg.training.num_negatives_train,
            alpha=model_cfg.alpha,
            loss_type=model_cfg.loss_type,
            device=device,
            log_interval=cfg.training.log_interval,
            epoch=epoch,
        )

        # Add Tensorboard scalar logs
        epoch_loss_total += loss_dict.get("loss", 0.0)
        writer.add_scalar("Loss/train", loss_dict.get("loss", 0.0), epoch)

        # Radius Tracking (Interval logging) - Max and Mean
        if hasattr(model, "entity_encoder") and hasattr(
            model.entity_encoder, "entity_rho"
        ):
            with torch.no_grad():
                radiuses = torch.nn.functional.softplus(
                    model.entity_encoder.entity_rho.weight
                )
                r_max = radiuses.max().item()
                r_mean = radiuses.mean().item()

                writer.add_scalar("Radius/Max", r_max, epoch)
                writer.add_scalar("Radius/Mean", r_mean, epoch)

                if epoch % cfg.training.log_interval == 0:
                    LOGGER.info(
                        "[ACTION] Epoch %d | Loss: %.4f | R_Max: %.4f | R_Mean: %.4f",
                        epoch,
                        loss_dict.get("loss", 0.0),
                        r_max,
                        r_mean,
                    )

        # Save periodically
        if epoch % 50 == 0 or epoch == (start_epoch + cfg.training.epochs - 1):
            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=dict(model_cfg),
                num_entities=dataset.num_entities,
                num_relations=dataset.num_relations,
            )
            LOGGER.info("[ACTION] Saved checkpoint block at Epoch %d", epoch)

    writer.close()


def run_train_init(cfg: DictConfig) -> None:
    run_training(cfg, resume=False)


def run_train(cfg: DictConfig) -> None:
    run_training(cfg, resume=True)
