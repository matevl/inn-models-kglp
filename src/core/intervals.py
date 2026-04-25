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


@dataclass
class ComplexInterval:
    """Interval in the complex plane representing a disc: c_re, c_im +/- r."""

    c_re: torch.Tensor
    c_im: torch.Tensor
    r: torch.Tensor

    def distance(self, other: "ComplexInterval") -> torch.Tensor:
        """Computes the distance between two complex intervals.

        The distance is defined as the sum of their radii minus the Euclidean
        distance between their centers.
        """
        diff_re = self.c_re - other.c_re
        diff_im = self.c_im - other.c_im
        dist_c = torch.sqrt(diff_re**2 + diff_im**2).sum(dim=-1)
        sum_r = (self.r + other.r).sum(dim=-1)
        return sum_r - dist_c


def irotate(h: ComplexInterval, r_phase: torch.Tensor, r_radius: torch.Tensor) -> ComplexInterval:
    """Rotates a ComplexInterval by a given phase and adds uncertainty.

    Args:
        h (ComplexInterval): The head entity complex interval (disc).
        r_phase (torch.Tensor): The rotation angle (phase) of the relation.
        r_radius (torch.Tensor): The uncertainty added by the relation.

    Returns:
        ComplexInterval: The rotated complex interval.
    """
    cos_r = torch.cos(r_phase)
    sin_r = torch.sin(r_phase)

    pred_c_re = h.c_re * cos_r - h.c_im * sin_r
    pred_c_im = h.c_re * sin_r + h.c_im * cos_r

    pred_radius = h.r + r_radius

    return ComplexInterval(pred_c_re, pred_c_im, pred_radius)

