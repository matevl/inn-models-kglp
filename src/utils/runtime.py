from __future__ import annotations
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig

_RUN_DIR: Path | None = None
_RUN_NAME: str | None = None
_TB_WRITER = None


class _ActionOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "[ACTION]" in msg


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("inn-models-kglp")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    try:
        from rich.logging import RichHandler

        handler = RichHandler(show_path=False)
        formatter = logging.Formatter("%(message)s")
    except Exception:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def start_run_logging(cfg: DictConfig, logs_root: str = "logs") -> Path:
    global _RUN_DIR, _RUN_NAME
    logger = configure_logging()
    if _RUN_DIR is not None:
        return _RUN_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = cfg.get("mode", "train")
    model_name = cfg.model.name if "model" in cfg and "name" in cfg.model else "unknown"
    dataset_name = (
        cfg.dataset.name if "dataset" in cfg and "name" in cfg.dataset else "unk_data"
    )

    _RUN_NAME = f"{timestamp}_{mode}_{model_name}_{dataset_name}"
    run_dir = Path(logs_root) / _RUN_NAME
    suffix = 1
    while run_dir.exists():
        _RUN_NAME = f"{timestamp}_{suffix}_{mode}_{model_name}"
        run_dir = Path(logs_root) / _RUN_NAME
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%m/%d/%y %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    for handler in logger.handlers:
        if handler is file_handler:
            continue
        handler.addFilter(_ActionOnlyFilter())

    _RUN_DIR = run_dir
    logger.info("[ACTION] Logging initialized in %s for mode '%s'", run_dir, mode)
    return run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, Any],
    num_entities: int,
    num_relations: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "num_entities": num_entities,
            "num_relations": num_relations,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def setup_tensorboard(tensorboard_dir: str | None, run_name: str | None = None) -> Any:
    global _TB_WRITER
    if _TB_WRITER is not None:
        return _TB_WRITER
    from torch.utils.tensorboard import SummaryWriter

    if not tensorboard_dir:
        tensorboard_dir = "runs"
    if run_name is None:
        run_name = _RUN_NAME if _RUN_NAME else "default_run"
    _TB_WRITER = SummaryWriter(log_dir=f"{tensorboard_dir}/{run_name}")
    return _TB_WRITER
