"""Shared tensor estimate interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Estimate:
    """Estimate stores a tensor value and its nonnegative standard error.

    ``stderr`` satisfies the invariant ``stderr >= 0``. The ``snr`` property
    represents zero reported noise as positive infinity, independently of
    ``value``.
    """

    value: torch.Tensor
    stderr: torch.Tensor

    @property
    def snr(self) -> torch.Tensor:
        """Return ``abs(value) / stderr``, or infinity where ``stderr`` is zero."""
        return torch.where(
            self.stderr > 0,
            torch.abs(self.value) / self.stderr,
            torch.inf,
        )
