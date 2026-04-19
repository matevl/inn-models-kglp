import torch
from torch.utils.data import DataLoader, TensorDataset


def test_adaptive_dataloader():
    # Simple dataset mimicking triplets (head, relation, tail)
    triplets = torch.randint(0, 100, (100, 3))
    dataset = TensorDataset(triplets)

    loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 1
    assert batch[0].shape == (10, 3)
