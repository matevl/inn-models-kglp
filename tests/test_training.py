import time
import pytest
import torch
import os

from src.core.training import train_epoch, create_train_dataloader
from src.models import build_link_predictor
from src.data.dataset import load_dataset


@pytest.mark.parametrize("model_type", ["inn_ours_mlp", "inn_lightgcn"])
def test_training_epoch_speed(model_type):
    dataset_path = "datasets/fb15k-237"
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found at {dataset_path}")

    dataset = load_dataset(dataset_path)

    num_entities = dataset.num_entities
    num_relations = dataset.num_relations
    train_triples = dataset.train
    batch_size = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_link_predictor(
        model_type=model_type,
        num_entities=num_entities,
        num_relations=num_relations,
        dim=200,
        gamma_margin=1.0,
        init_rho=-5.0,
        hidden_layers=[800] if model_type == "inn_ours_mlp" else None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if device.type == "cuda":
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = True
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Failed to compile model: {e}")

    # Note: Using 0 workers for tests to avoid multiprocessing overhead and hanging
    t0 = time.time()

    loader = create_train_dataloader(
        train_triples=train_triples,
        batch_size=batch_size,
        device=torch.device("cpu"),  # Use CPU device for dataloader workers in tests
    )

    train_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        gamma_margin=1.0,
        num_entities=num_entities,
        num_negatives=32,
        alpha=1.0,
        loss_type="adversarial",
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
        duration = time.time() - t0
        assert duration < 5.0, (
            f"Training epoch for {model_type} took {duration:.2f}s (>= 5s on GPU)"
        )
    else:
        duration = time.time() - t0
        assert duration < 30.0, (
            f"Training epoch for {model_type} took {duration:.2f}s (>= 30s on CPU)"
        )
