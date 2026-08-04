"""Shared tensor estimate interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class Estimate:
    """Estimate stores a tensor value and its nonnegative standard error.

    ``stderr`` satisfies the invariant ``stderr >= 0``. The derived ``snr`` field
    represents zero reported noise as positive infinity, independently of
    ``value``.
    """

    value: torch.Tensor
    stderr: torch.Tensor
    snr: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        """Derive ``snr`` once from ``value`` and ``stderr`` after initialization."""
        object.__setattr__(
            self,
            "snr",
            torch.where(
                self.stderr > 0,
                torch.abs(self.value) / self.stderr,
                torch.inf,
            ),
        )
