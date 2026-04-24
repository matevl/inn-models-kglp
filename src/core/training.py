"""Training loop and utilities."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from models import LOSS_TYPE


def sample_negative_triples(
    pos_triplets: torch.Tensor,
    num_entities: int,
    num_negatives: int = 64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample negative triples by corrupting head or tail (Fully Vectorized).

    Args:
        pos_triplets (torch.Tensor): A tensor of shape (batch_size, 3) containing positive triples (head, relation, tail).
        num_entities (int): The total number of entities in the knowledge graph.
        num_negatives (int, optional): The number of negative samples to generate per positive triple. Defaults to 64.
        device (torch.device | None, optional): The device on which to perform sampling. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_negatives, 3) containing the sampled negative triples.
    """
    bsz = pos_triplets.size(0)

    # Pre-allocate tensor to optimize memory allocation
    neg = pos_triplets.new_empty((bsz, num_negatives, 3))

    # Broadcast relation indices
    neg[:, :, 1] = pos_triplets[:, 1:2]

    # Generate random entities and boolean mask
    if device is None:
        device = torch.device("cpu")
    rand_ents = torch.randint(0, num_entities, (bsz, num_negatives), device=device)
    replace_head = torch.rand((bsz, num_negatives), device=device) < 0.5

    # Apply masking via torch.where to minimize CUDA synchronization
    neg[:, :, 0] = torch.where(replace_head, rand_ents, pos_triplets[:, 0:1])
    neg[:, :, 2] = torch.where(replace_head, pos_triplets[:, 2:3], rand_ents)

    return neg


def create_train_collate_fn():
    """Creates a simple collate_fn returning only positive batches. Sampling is done on GPU."""

    def collate_fn(batch):
        return default_collate(batch)

    return collate_fn


def create_train_dataloader(
    train_triples: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    """Creates an optimized DataLoader for training.

    Args:
        train_triples (torch.Tensor): A tensor of shape (num_triples, 3) representing the training dataset.
        batch_size (int): The number of triples per batch.
        device (torch.device): The device used for training, used to configure pinned memory and workers.

    Returns:
        DataLoader: A PyTorch DataLoader configured for the training data.
    """
    import multiprocessing

    available_workers = (
        multiprocessing.cpu_count() if multiprocessing.cpu_count() else 1
    )
    num_workers = min(available_workers, 8) if device.type != "cpu" else 0
    pin_memory = device.type == "cuda"

    return DataLoader(
        train_triples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=create_train_collate_fn(),
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float,
    num_entities: int,
    num_negatives: int,
    writer=None,
    epoch: int = 0,
    alpha: float = 1.0,
    loss_type: str = "self_adversarial",
    log_interval: int = 10,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, list[dict]]:
    """Train the model for one epoch, optionally logging per-iteration metrics.

    Args:
        model (torch.nn.Module): The link prediction model to train.
        loader (DataLoader): The DataLoader providing training batches.
        optimizer (torch.optim.Optimizer): The optimizer used for weight updates.
        device (torch.device): The device on which computations are performed.
        margin (float): The margin parameter used in the loss function.
        num_entities (int): Total number of entities in the dataset for negative sampling.
        num_negatives (int): Number of negative samples per positive triple.
        writer (Any, optional): TensorBoard SummaryWriter for logging. Defaults to None.
        epoch (int, optional): The current epoch number. Defaults to 0.
        alpha (float, optional): The alpha temperature parameter for adversarial sampling. Defaults to 1.0.
        loss_type (str, optional): The type of loss function to use (e.g., 'adversarial'). Defaults to "adversarial".
        log_interval (int, optional): Log metrics every N iterations to reduce overhead. Defaults to 10.
        scaler (torch.amp.GradScaler | None, optional): Gradient scaler for mixed precision training. Defaults to None.

    Returns:
        tuple[float, list[dict]]: A tuple containing:
            - The average loss over the epoch.
            - A list of iteration metrics dictionaries with keys 'iteration', 'loss', 'accuracy', and 'batch_items'.
    """
    model.train()

    total_loss = 0.0
    total_items = 0
    iteration_metrics = []

    for batch_idx, pos_batch in enumerate(loader):
        pos_batch = pos_batch.to(device, non_blocking=True)
        # Vectorized negative sampling on device
        neg_batch = sample_negative_triples(
            pos_batch,
            num_entities=num_entities,
            num_negatives=num_negatives,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            pos_scores, neg_scores = model(pos_batch, neg_batch)
            loss_fn = LOSS_TYPE.get(loss_type, LOSS_TYPE["self_adversarial"])
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

        iteration_metrics.append(
            {
                "iteration": batch_idx,
                "loss_tensor": batch_loss_tensor,
                "acc_tensor": acc_tensor,
                "batch_items": batch_items,
            }
        )

    # Defer .item() calls to prevent GPU synchronization stalls
    if device.type == "cuda":
        torch.cuda.synchronize()

    for metrics in iteration_metrics:
        if "accuracy" in metrics:
            continue

        # Extract scalars from tensors
        b_loss = metrics["loss_tensor"].item()
        b_acc = metrics["acc_tensor"].item()
        b_items = metrics["batch_items"]

        metrics["loss"] = b_loss
        metrics["accuracy"] = b_acc

        total_loss += b_loss * b_items

        if writer is not None:
            global_step = epoch * len(loader) + metrics["iteration"]
            writer.add_scalar("Loss/train_iteration", b_loss, global_step)
            writer.add_scalar("Accuracy/train_iteration", b_acc, global_step)

    avg_loss = total_loss / total_items if total_items > 0 else 0.0

    return avg_loss, iteration_metrics
