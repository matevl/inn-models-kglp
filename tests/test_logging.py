import logging
import torch
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch
from pathlib import Path

from src.data.dataset import KGDataset
from src.train import run_training
import src.utils.runtime as runtime


@pytest.fixture
def small_dataset():
    num_entities = 100
    num_relations = 10

    # Create 32 mock triples so batching works fine
    h = torch.randint(0, num_entities, (32,))
    r = torch.randint(0, num_relations, (32,))
    t = torch.randint(0, num_entities, (32,))
    train_triples = torch.stack([h, r, t], dim=1)

    return KGDataset(
        dataset_path=Path("mock_path"),
        train=train_triples,
        valid=train_triples.clone(),
        test=train_triples.clone(),
        entity_to_id={str(i): i for i in range(num_entities)},
        relation_to_id={str(i): i for i in range(num_relations)},
    )


def test_10_epoch_logging_small_dataset_mlp_layers(small_dataset, caplog, tmp_path):
    caplog.set_level(logging.INFO, logger="inn-models-kglp")

    # MLP with 2 hidden layer -> [2, 2] per the user prompt ("with 2 [] hidden layer -> 2" implies `hidden_layers: [2, 2]`, or simply `[2]` for "a 2 hidden layer". I'll configure them as `[2]`).
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "device": "cpu",
            "tensorboard_dir": str(tmp_path / "runs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint": "test_ckpt.pt",
            "dataset": {"name": "mock_small", "path": "mock_path"},
            "model": {
                "name": "inn_ours_mlp",
                "dim": 16,
                "gamma_margin": 1.0,
                "init_rho": -5.0,
                "hidden_layers": [2],
                "alpha": 1.0,
                "loss_type": "adversarial",
            },
            "training": {
                "epochs": 10,
                "batch_size": 8,
                "lr": 0.001,
                "num_negatives_train": 4,
                "log_interval": 10,
            },
        }
    )

    runtime._RUN_NAME = "test_logging_run"

    with patch("src.train.load_dataset", return_value=small_dataset):
        run_training(cfg, resume=False)

    records = caplog.records

    # Verify [RECAP] log strings exist
    recap_logs = [r for r in records if "[RECAP]" in r.getMessage()]

    assert len(recap_logs) == 1, (
        "Recap should be printed exactly once since log_interval=10 and epochs=10"
    )

    # Verify the recap contains the metrics of the 10th epoch
    assert "Epoch 10 finished" in recap_logs[0].getMessage()
