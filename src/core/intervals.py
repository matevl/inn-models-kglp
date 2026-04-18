from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Interval:
    """Interval in Center-Radius form: x in [c +/- r]."""

    c: torch.Tensor
    r: torch.Tensor

    def lu(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.c - self.r, self.c + self.r

    @staticmethod
    def from_lu(lo: torch.Tensor, hi: torch.Tensor) -> "Interval":
        c = (lo + hi) / 2
        r = (hi - lo) / 2
        return Interval(c, r)

    def clip_nonnegative_radius(self) -> "Interval":
        self.r = torch.clamp(self.r, min=0)
        return self


def iadd(a: Interval, b: Interval) -> Interval:
    return Interval(a.c + b.c, a.r + b.r)


def isub(a: Interval, b: Interval) -> Interval:
    return Interval(a.c - b.c, a.r + b.r)


def iscale(alpha: Any, x: Interval) -> Interval:
    abs_alpha = torch.abs(alpha) if isinstance(alpha, torch.Tensor) else abs(alpha)
    return Interval(alpha * x.c, abs_alpha * x.r)


def ilinear(W: torch.Tensor, x: Interval) -> Interval:
    """y = W x with interval bound y in [W c +/- |W| r]."""
    absW = torch.abs(W)
    yc = torch.matmul(W, x.c)
    yr = torch.matmul(absW, x.r)
    return Interval(yc, yr)


def interval_relu(x: Interval) -> Interval:
    """Conservative ReLU bounds from [lo, hi] endpoints."""
    lo, hi = x.lu()
    return Interval.from_lu(F.relu(lo), F.relu(hi))


def interval_sigmoid(x: Interval) -> Interval:
    """Sigmoid is monotone increasing, so endpoints bound the image."""
    lo, hi = x.lu()
    return Interval.from_lu(torch.sigmoid(lo), torch.sigmoid(hi))


def interval_tanh(x: Interval) -> Interval:
    lo, hi = x.lu()
    return Interval.from_lu(torch.tanh(lo), torch.tanh(hi))


def interval_logsumexp(x: Interval, axis: int = -1) -> Interval:
    """Simple logsumexp bounds via endpoint monotonicity."""
    lo, hi = x.lu()
    lo_s = torch.logsumexp(lo, dim=axis)
    hi_s = torch.logsumexp(hi, dim=axis)
    return Interval.from_lu(lo_s, hi_s)
