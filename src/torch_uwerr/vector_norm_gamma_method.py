"""Widehat-first scalar vector-norm gamma-method estimates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torch_uwerr.estimate import Estimate
from torch_uwerr.gamma_method import gamma_method_mean


@dataclass(frozen=True, slots=True)
class VectorNormGammaMethodEstimate(Estimate):
    """VectorNormGammaMethodEstimate holds norm estimates and Q-bar diagnostics."""

    stderr_of_stderr: torch.Tensor
    Q_bar_tau_int: torch.Tensor
    replica_count: int
    widehat_count: int
    widecheck_count: int
    sample_count: int
    Q_bar_autocovariance: torch.Tensor | None
    Q_bar_autocorrelation: torch.Tensor | None


def vector_norm_gamma_method(
    Q_history: torch.Tensor,
    *,
    replica_count: int,
    widehat_count: int,
    gamma_method_s: float = 2.0,
    accumulation_dtype: torch.dtype = torch.float64,
    return_autocorrelation: bool = False,
) -> VectorNormGammaMethodEstimate:
    """Return the norm estimate induced by a scalar ``Q_i`` history.

    ``Q_history`` has shape ``(*batch, sample_count)`` and stores
    ``Q_i = widehat_overline_X dot widecheck_X_i``. ``replica_count`` is the
    total replica count ``R`` and ``widehat_count`` is the first-block count
    ``M``; the returned ``widecheck_count`` is ``R - M``. The returned
    ``value`` is ``sign(Q_bar) * sqrt(abs(Q_bar))``. The gamma-method standard
    error of ``Q_bar`` is scaled to the corresponding ``|X|^2`` standard error
    before ``stderr`` propagates it through the signed square root.
    """
    Q_tensor = torch.as_tensor(Q_history)
    Q = gamma_method_mean(
        Q_tensor.unsqueeze(-2),
        gamma_method_s=gamma_method_s,
        accumulation_dtype=accumulation_dtype,
        return_autocorrelation=return_autocorrelation,
    )
    R = replica_count
    M = widehat_count
    K = R - M
    norm_sq = Q.value
    # The Q-history gamma estimate gives the constant-free leading variance;
    # the full Q_bar variance is larger by R / M.
    norm_sq_stderr_scale = 2 * K**0.5 / R**0.5
    norm_sq_stderr = norm_sq_stderr_scale * Q.stderr
    norm_sq_stderr_of_stderr = norm_sq_stderr_scale * Q.stderr_of_stderr
    norm = torch.sign(norm_sq) * torch.sqrt(torch.abs(norm_sq))
    norm_magnitude = torch.abs(norm)
    return VectorNormGammaMethodEstimate(
        value=norm,
        stderr=0.5 * norm_sq_stderr / norm_magnitude,
        stderr_of_stderr=0.5 * norm_sq_stderr_of_stderr / norm_magnitude,
        Q_bar_tau_int=Q.tau_int,
        replica_count=R,
        widehat_count=M,
        widecheck_count=K,
        sample_count=Q_tensor.shape[-1],
        Q_bar_autocovariance=Q.autocovariance,
        Q_bar_autocorrelation=Q.autocorrelation,
    )


class VectorNormGammaMethodHelper:
    """Wrapper class for the underlying vector norm gamma method computation

    This class enforces the most common workflow for using vector_norm_gamma_method:
    1. form X_overline_widehat
    2. use X_overline_widehat to form Q_i = X_overline_widehat * X_widecheck_i
    3. call vector_norm_gamma_method on the Q_i tensor
    """

    __slots__ = (
        "accumulation_dtype",
        "widehat_count",
        "widecheck_count",
        "replica_count",
        "sample_count",
        "_widehat",
        "_widehat_accumulator",
        "_widecheck_projection_accumulator",
    )

    def __init__(
        self,
        replica_count: int,
        sample_count: int,
        *,
        widehat_count: int | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize an ordered vector-norm estimator for ``replica_count`` and samples."""
        self.replica_count = replica_count
        self.sample_count = sample_count
        self.widehat_count = replica_count // 2 if widehat_count is None else widehat_count
        self.widecheck_count = replica_count - self.widehat_count
        self.accumulation_dtype = accumulation_dtype
        self._widehat_accumulator: torch.Tensor | None = None
        self._widehat: torch.Tensor | None = None
        self._widecheck_projection_accumulator: torch.Tensor | None = None

    @property
    def widehat(self) -> torch.Tensor:
        """Return the finalized ``widehat_overline_X`` covector."""
        if self._widehat is None:
            raise RuntimeError("finalize_widehat() must be called before accessing widehat")
        return self._widehat

    def accumulate_widehat_term(self, widehat_term: torch.Tensor) -> None:
        """Accumulate an unnormalized partial term for ``widehat_overline_X``.

        widehat_term contributes to ``sum_r overline{X}^r`` for any subset of the
        ``widehat_X`` replicas. finalize_widehat applies the missing ``1 / M``
        factor.
        """
        if self._widehat is not None:
            raise RuntimeError("widehat terms cannot be added after finalize_widehat()")
        x = torch.as_tensor(widehat_term).to(dtype=self.accumulation_dtype)
        if self._widehat_accumulator is None:
            self._widehat_accumulator = x.clone()
        else:
            self._widehat_accumulator = self._widehat_accumulator + x

    def finalize_widehat(self) -> torch.Tensor:
        """Store and return ``widehat_overline_X`` after applying ``1 / M``."""
        if self._widehat is not None:
            return self._widehat
        if self._widehat_accumulator is None:
            raise RuntimeError(
                "accumulate_widehat_term() must be called before finalize_widehat()"
            )
        self._widehat = self._widehat_accumulator / self.widehat_count
        self._widehat_accumulator = None
        return self._widehat

    def accumulate_widecheck_projection_term(
        self,
        projection_term: torch.Tensor,
    ) -> None:
        """Accumulate an unnormalized length-``sample_count`` projection term.

        projection_term contributes to
        ``sum_r widehat_overline_X dot widecheck_X_i^r``. compute applies the
        missing ``1 / K`` factor.
        """
        if self._widehat is None:
            raise RuntimeError(
                "finalize_widehat() must be called before adding widecheck projections"
            )
        Q_projection = torch.as_tensor(projection_term).to(dtype=self.accumulation_dtype)
        if Q_projection.shape != (self.sample_count,):
            raise ValueError("projection_term must have shape (sample_count,)")
        if self._widecheck_projection_accumulator is None:
            self._widecheck_projection_accumulator = Q_projection.clone()
        else:
            self._widecheck_projection_accumulator = (
                self._widecheck_projection_accumulator + Q_projection
            )

    def compute(
        self,
        *,
        gamma_method_s: float = 2.0,
        return_autocorrelation: bool = False,
    ) -> VectorNormGammaMethodEstimate:
        """Return the vector-norm estimate after applying the ``1 / K`` factor."""
        if self._widehat is None:
            raise RuntimeError("finalize_widehat() must be called before compute()")
        if self._widecheck_projection_accumulator is None:
            raise RuntimeError(
                "accumulate_widecheck_projection_term() must be called before compute()"
            )
        Q_history = self._widecheck_projection_accumulator / self.widecheck_count
        return vector_norm_gamma_method(
            Q_history,
            replica_count=self.replica_count,
            widehat_count=self.widehat_count,
            gamma_method_s=gamma_method_s,
            accumulation_dtype=self.accumulation_dtype,
            return_autocorrelation=return_autocorrelation,
        )
