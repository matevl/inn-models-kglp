"""Training loop and utilities."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models import LOSS_TYPE


def sample_negative_triples(
    pos_triplets: torch.Tensor,
    num_entities: int,
    num_negatives: int = 64,
    device: torch.device = None,
) -> torch.Tensor:
    """Sample negative triples by corrupting head or tail (Fully Vectorized)."""
    bsz = pos_triplets.size(0)

    # Pre-allocate empty tensor to avoid expensive copies and .repeat()
    neg = pos_triplets.new_empty((bsz, num_negatives, 3))

    # Broadcast standard values
    neg[:, :, 1] = pos_triplets[:, 1:2]

    # Generate random entities and boolean mask in one go
    rand_ents = torch.randint(0, num_entities, (bsz, num_negatives), device=device)
    replace_head = torch.rand((bsz, num_negatives), device=device) < 0.5

    # Use torch.where instead of boolean indexing to avoid CUDA synchronization
    neg[:, :, 0] = torch.where(replace_head, rand_ents, pos_triplets[:, 0:1])
    neg[:, :, 2] = torch.where(replace_head, pos_triplets[:, 2:3], rand_ents)

    return neg


def train_epoch(
    model: torch.nn.Module,
    train_triples: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    num_entities: int,
    margin: float,
    writer=None,
    epoch: int = 0,
    num_negatives: int = 64,
    alpha: float = 1.0,
    loss_type: str = "adversarial",
    log_interval: int = 10,
    scaler: torch.amp.GradScaler = None,
) -> tuple[float, list[dict]]:
    """
    Train for one epoch, optionally logging per-iteration metrics.

    Returns:
            Tuple of (average_loss, iteration_metrics)
            where iteration_metrics is a list of dicts with keys: iteration, loss, accuracy

    Args:
            log_interval: Log metrics every N iterations to reduce overhead (default: 10)
    """
    model.train()

    # Adaptive dataloader num_workers for laptop compatibility vs A40 power
    import multiprocessing

    # Uses 0 if cpu count is 1 or None. Limits to 8 max to avoid memory overhead.
    available_workers = (
        multiprocessing.cpu_count() if multiprocessing.cpu_count() else 1
    )
    num_workers = min(available_workers, 8) if device.type != "cpu" else 0
    pin_memory = device.type == "cuda"

    # Optimized DataLoader: async loading + pinned memory for GPU transfer pipelining
    loader = DataLoader(
        train_triples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    total_loss = 0.0
    total_items = 0
    iteration_metrics = []

    for batch_idx, pos_batch in enumerate(loader):
        pos_batch = pos_batch.to(device, non_blocking=True)
        neg_batch = sample_negative_triples(
            pos_batch,
            num_entities=num_entities,
            num_negatives=num_negatives,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            pos_scores, neg_scores = model(pos_batch, neg_batch)
            loss_fn = LOSS_TYPE.get(loss_type, LOSS_TYPE["adversarial"])
            loss = loss_fn(pos_scores, neg_scores, margin=margin, alpha=alpha)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_items = pos_batch.size(0)
        total_items += batch_items

        batch_loss_tensor = loss.detach()

        with torch.no_grad():
            acc_tensor = torch.mean((pos_scores.unsqueeze(-1) > neg_scores).float())

        is_log_step = batch_idx % log_interval == 0 or batch_idx == len(loader) - 1
        radii_stats = model.get_radii_stats() if is_log_step else None

        iteration_metrics.append(
            {
                "iteration": batch_idx,
                "loss_tensor": batch_loss_tensor,
                "acc_tensor": acc_tensor,
                "radii": radii_stats,
                "batch_items": batch_items,
            }
        )

        if is_log_step:
            last_radii = radii_stats
            for i in range(
                max(0, len(iteration_metrics) - log_interval), len(iteration_metrics)
            ):
                metrics = iteration_metrics[i]

                if "accuracy" in metrics:
                    continue

                b_loss = metrics["loss_tensor"].item()
                b_acc = metrics["acc_tensor"].item()
                r_stats = metrics["radii"] or last_radii
                b_items = metrics["batch_items"]

                metrics["loss"] = b_loss
                metrics["accuracy"] = b_acc
                metrics["entity_r_mean"] = r_stats["entity_r_mean"]
                metrics["entity_r_max"] = r_stats["entity_r_max"]
                metrics["rel_r_mean"] = r_stats["rel_r_mean"]
                metrics["rel_r_max"] = r_stats["rel_r_max"]

                total_loss += b_loss * b_items

                if writer is not None:
                    global_step = epoch * len(loader) + metrics["iteration"]
                    writer.add_scalar("Loss/train_iteration", b_loss, global_step)
                    writer.add_scalar("Accuracy/train_iteration", b_acc, global_step)
                    for k, v in r_stats.items():
                        writer.add_scalar(f"Radii_iteration/{k}", v, global_step)

    if total_items == 0:
        return 0.0, iteration_metrics
    return total_loss / total_items, iteration_metrics
