import torch
from src.core.intervals import Interval, interval_relu


def test_interval_initialization():
    center = torch.tensor([[1.0, -1.0]])
    radius = torch.tensor([[0.5, 0.5]])
    interv = Interval(center, radius)

    assert torch.allclose(interv.c, center)
    assert torch.allclose(interv.r, radius)


def test_interval_relu():
    # Center = [1.0, -1.0], Radius = [0.5, 0.5]
    # Bound 1: [0.5, 1.5] -> ReLU -> [0.5, 1.5] -> c=1.0, r=0.5
    # Bound 2: [-1.5, -0.5] -> ReLU -> [0.0, 0.0] -> c=0.0, r=0.0
    center = torch.tensor([[1.0, -1.0]])
    radius = torch.tensor([[0.5, 0.5]])
    interv = Interval(center, radius)

    out = interval_relu(interv)

    expected_c = torch.tensor([[1.0, 0.0]])
    expected_r = torch.tensor([[0.5, 0.0]])

    assert torch.allclose(out.c, expected_c)
    assert torch.allclose(out.r, expected_r)
