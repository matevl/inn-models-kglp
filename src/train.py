from __future__ import annotations
import logging
from pathlib import Path
from omegaconf import DictConfig

import torch

from core.training import train_epoch, create_train_dataloader
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
    model = torch.compile(model)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    epoch_loss_total = 0.0
    log_buffer = []

    train_loader = create_train_dataloader(
        train_triples=dataset.train,
        batch_size=cfg.training.batch_size,
        device=device,
    )

    # Custom training loop to show logs dynamically
    for epoch in range(start_epoch, start_epoch + cfg.training.epochs):
        avg_loss, iter_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            margin=model_cfg.margin,
            num_entities=dataset.num_entities,
            num_negatives=cfg.training.num_negatives_train,
            alpha=model_cfg.alpha,
            loss_type=model_cfg.loss_type,
            log_interval=cfg.training.log_interval,
            epoch=epoch,
            scaler=scaler,
        )

        # Add Tensorboard scalar logs
        epoch_loss_total += avg_loss
        writer.add_scalar("Loss/train", avg_loss, epoch)

        logMsg = f"[EPOCH] Epoch {epoch} finished | train_loss={avg_loss:.6f}"

        # Calculate epoch accuracy if present
        if iter_metrics and "accuracy" in iter_metrics[(0)]:
            avg_acc = sum(m["accuracy"] * m["batch_items"] for m in iter_metrics) / sum(
                m["batch_items"] for m in iter_metrics
            )
            logMsg += f" | train_accuracy={avg_acc:.4f}"
            writer.add_scalar("Accuracy/train", avg_acc, epoch)

        LOGGER.info(logMsg)
        log_buffer.append(logMsg)

        # Radius Tracking (Interval logging) - Max, Min and Mean
        # Only compute these expensive metrics at the log interval
        if epoch % cfg.training.log_interval == 0:
            if hasattr(model, "entity_encoder") and hasattr(
                model.entity_encoder, "entity_rho"
            ):
                with torch.no_grad():
                    radiuses = torch.nn.functional.softplus(
                        model.entity_encoder.entity_rho.weight
                    )
                    r_max = radiuses.max().item()
                    r_min = radiuses.min().item()
                    r_mean = radiuses.mean().item()

                    writer.add_scalar("Radius/Max", r_max, epoch)
                    writer.add_scalar("Radius/Min", r_min, epoch)
                    writer.add_scalar("Radius/Mean", r_mean, epoch)

                    log_buffer.append(
                        f"  -> R_Max: {r_max:.4f} | R_Min: {r_min:.4f} | R_Mean: {r_mean:.4f}"
                    )

                    # Log radius metrics
                    LOGGER.info(
                        "[ACTION] Epoch %d Radius -> R_Max: %.4f | R_Min: %.4f | R_Mean: %.4f",
                        epoch,
                        r_max,
                        r_min,
                        r_mean,
                    )

            recap_str = "\n\t".join(log_buffer)
            LOGGER.info("[ACTION] Recap of last epochs:\n\t%s", recap_str)
            log_buffer.clear()

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
