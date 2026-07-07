"""Dense torch implementation of pyerrors-style gamma-method means."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GammaMethodEstimate: # to be renamed in future commit
    """Batch-shaped mean and gamma-method uncertainty tensors."""

    value: torch.Tensor
    stderr: torch.Tensor
    # snr: torch.Tensor # The is just value/stderr, and should be added for convenience
    tau_int: torch.Tensor
    stderr_of_stderr: torch.Tensor | None
    window: torch.Tensor | None
    autocovariance: torch.Tensor | None = None # Γ(t)
    autocorrelation: torch.Tensor | None = None # ⍴(t)


def _as_dense_chains(
    samples: torch.Tensor,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    """Return samples as a floating tensor shaped as (*batch, chains, samples)."""
    x = torch.as_tensor(samples).to(dtype=accumulation_dtype)
    if x.ndim == 1:
        return x.reshape(1, x.shape[0])
    return x


def _pooled_autocovariance(centered: torch.Tensor, w_max: int) -> torch.Tensor:
    """Return pooled dense-chain autocovariance for lags [0, w_max)."""
    sample_count = centered.shape[-1]
    chain_count = centered.shape[-2]
    padding = sample_count + w_max + (sample_count + w_max) % 2
    spectrum = torch.fft.rfft(centered, n=padding, dim=-1)
    lag_sums = torch.fft.irfft(spectrum.abs().square(), n=padding, dim=-1)[..., :w_max]
    pair_counts = torch.arange(
        sample_count,
        sample_count - w_max,
        -1,
        dtype=centered.dtype,
        device=centered.device,
    )
    return lag_sums.sum(dim=-2) / (chain_count * pair_counts)


def _safe_autocorrelation(autocovariance: torch.Tensor) -> torch.Tensor:
    """Return autocorrelation while keeping zero-variance histories finite."""
    variance = autocovariance[..., :1]
    return torch.where(
        variance != 0,
        autocovariance / variance,
        torch.zeros_like(autocovariance),
    )


def _tau_summation_history(autocorrelation: torch.Tensor) -> torch.Tensor:
    """Return pyerrors cumulative tau_int history for autocorrelation lags."""
    leading_half = torch.full_like(autocorrelation[..., :1], 0.5)
    τ = torch.cumsum(
        torch.cat((leading_half, autocorrelation[..., 1:]), dim=-1),
        dim=-1,
    )
    return torch.maximum(τ, torch.full_like(τ, 0.5 + torch.finfo(τ.dtype).eps))


def _automatic_window(
    τ_history: torch.Tensor,
    total_count: int,
    gamma_method_s: float,
) -> torch.Tensor:
    """Return pyerrors automatic-window indices for τ_history and S."""
    w_max = τ_history.shape[-1]
    if w_max <= 1:
        return torch.zeros(τ_history.shape[:-1], dtype=torch.long, device=τ_history.device)

    candidates = torch.arange(1, w_max, dtype=τ_history.dtype, device=τ_history.device)
    τ_tail = gamma_method_s / torch.log(
        (2 * τ_history[..., 1:] + 1) / (2 * τ_history[..., 1:] - 1)
    )
    g_w = torch.exp(-candidates / τ_tail) - τ_tail / torch.sqrt(candidates * total_count)
    below_zero = g_w < 0
    below_zero[..., -1] = True
    return below_zero.to(torch.long).argmax(dim=-1) + 1


def _gather_lag(history: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Return history entries selected by the per-batch window tensor."""
    return torch.gather(history, dim=-1, index=window.unsqueeze(-1)).squeeze(-1)


def gamma_method_mean(
    samples: torch.Tensor,
    *,
    gamma_method_s: float = 2.0,
    accumulation_dtype: torch.dtype = torch.float64,
    return_autocorrelation: bool = False,
) -> GammaMethodEstimate:
    """Estimate a dense-history mean and pyerrors-style gamma-method error.

    samples is interpreted as (..., chains, samples), except a one-dimensional
    input is treated as one unbatched chain. The returned value, stderr, tau_int,
    stderr_of_stderr, and window tensors have the leading batch shape, while the
    optional autocovariance and autocorrelation tensors have shape
    (*batch_shape, N // 2).
    """
    x = _as_dense_chains(samples, accumulation_dtype)
    sample_count = x.shape[-1]
    chain_count = x.shape[-2]
    total_count = chain_count * sample_count
    w_max = sample_count // 2

    value = x.mean(dim=(-2, -1))
    centered = x - x.mean(dim=-1, keepdim=True)
    autocovariance = _pooled_autocovariance(centered, w_max)
    autocorrelation = _safe_autocorrelation(autocovariance)
    variance = autocovariance[..., 0]
    zero_variance = variance == 0

    if gamma_method_s == 0.0:
        window = torch.zeros_like(variance, dtype=torch.long)
        tau_int = torch.full_like(variance, 0.5)
        stderr = torch.sqrt(variance / (total_count - 1))
        stderr_of_stderr = stderr * torch.sqrt(
            torch.as_tensor(0.5 / total_count, dtype=x.dtype, device=x.device)
        )
    else:
        τ_history = _tau_summation_history(autocorrelation)
        window = _automatic_window(τ_history, total_count, gamma_method_s)
        window_float = window.to(dtype=x.dtype)
        τ_window = _gather_lag(τ_history, window)
        bias = (1 + (2 * window_float + 1) / total_count) / (1 + 1 / total_count)
        tau_int = τ_window * bias
        stderr = torch.sqrt(2 * tau_int * variance * (1 + 1 / total_count) / total_count)
        stderr_of_stderr = stderr * torch.sqrt((window_float + 0.5) / total_count)

    tau_int = torch.where(zero_variance, torch.full_like(tau_int, 0.5), tau_int)
    stderr = torch.where(zero_variance, torch.zeros_like(stderr), stderr)
    stderr_of_stderr = torch.where(
        zero_variance,
        torch.zeros_like(stderr_of_stderr),
        stderr_of_stderr,
    )
    window = torch.where(zero_variance, torch.zeros_like(window), window)

    if return_autocorrelation:
        returned_autocovariance = autocovariance
        returned_autocorrelation = autocorrelation
    else:
        returned_autocovariance = None
        returned_autocorrelation = None

    return GammaMethodEstimate(
        value=value,
        stderr=stderr,
        tau_int=tau_int,
        stderr_of_stderr=stderr_of_stderr,
        window=window,
        autocovariance=returned_autocovariance,
        autocorrelation=returned_autocorrelation,
    )


class GammaMethodMeanAccumulator:
    """Accumulate gamma-method mean estimates over complete replica chunks."""

    # __slots__ limits GammaMethodMeanAccumulator to configuration and aggregate state.
    __slots__ = (
        "accumulation_dtype",
        "batch_shape",
        "gamma_method_s",
        "replica_count",
        "sample_count",
        "_stderr_sq_sum",
        "_tau_int_sum",
        "_value_sum",
    )

    def __init__(
        self,
        *,
        gamma_method_s: float = 2.0,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize empty weighted aggregates for ``gamma_method_s`` and dtype."""
        self.gamma_method_s = gamma_method_s
        self.accumulation_dtype = accumulation_dtype
        self.replica_count = 0
        self.sample_count: int | None = None
        self.batch_shape: torch.Size | None = None
        self._value_sum: torch.Tensor | None = None
        self._stderr_sq_sum: torch.Tensor | None = None
        self._tau_int_sum: torch.Tensor | None = None

    def accumulate(self, chunk: torch.Tensor) -> None:
        """Add one complete-replica ``chunk`` with shape ``(*batch, R, N)``.

        ``chunk`` is normalized with _as_dense_chains before shape metadata is
        checked, so one-dimensional chunks are treated as one replica. Only
        weighted value, stderr, and tau_int aggregates are retained.
        """
        x = _as_dense_chains(chunk, self.accumulation_dtype)
        chunk_batch_shape = x.shape[:-2]
        chunk_replica_count = x.shape[-2]
        chunk_sample_count = x.shape[-1]

        is_first_chunk = self.batch_shape is None
        if (
            not is_first_chunk
            and (
                chunk_batch_shape != self.batch_shape
                or chunk_sample_count != self.sample_count
            )
        ):
            raise ValueError(
                "chunk must match the existing batch shape and sample_count"
            )

        estimate = gamma_method_mean(
            x,
            gamma_method_s=self.gamma_method_s,
            accumulation_dtype=self.accumulation_dtype,
        )
        if is_first_chunk:
            self.batch_shape = chunk_batch_shape
            self.sample_count = chunk_sample_count

        weighted_value = chunk_replica_count * estimate.value
        weighted_stderr = chunk_replica_count * estimate.stderr
        weighted_tau_int = chunk_replica_count * estimate.tau_int

        if self._value_sum is None:
            self._value_sum = weighted_value.detach()
            self._stderr_sq_sum = weighted_stderr.square().detach()
            self._tau_int_sum = weighted_tau_int.detach()
        else:
            self._value_sum += weighted_value.detach()
            self._stderr_sq_sum += weighted_stderr.square().detach()
            self._tau_int_sum += weighted_tau_int.detach()
        self.replica_count += chunk_replica_count

    def result(self) -> GammaMethodEstimate:
        """Return replica-weighted means and omit history-dependent diagnostics."""
        if (
            self.replica_count == 0
            or self._value_sum is None
            or self._stderr_sq_sum is None
            or self._tau_int_sum is None
        ):
            raise RuntimeError("accumulate() must be called before result()")

        R = self.replica_count
        return GammaMethodEstimate(
            value=self._value_sum / R,
            stderr=torch.sqrt(self._stderr_sq_sum) / R,
            tau_int=self._tau_int_sum / R,
            stderr_of_stderr=None,
            window=None,
            autocovariance=None,
            autocorrelation=None,
        )
