"""Dense torch implementation of pyerrors-style gamma-method means."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GammaMethodEstimate: # to be renamed in future commit
    """Batch-shaped gamma-method estimates with per-replica sample lengths."""

    value: torch.Tensor
    stderr: torch.Tensor
    tau_int: torch.Tensor
    sample_shapes: torch.Tensor
    stderr_of_stderr: torch.Tensor | None
    window: torch.Tensor | None
    autocovariance: torch.Tensor | None = None # Γ(t)
    autocorrelation: torch.Tensor | None = None # ⍴(t)

    @property
    def snr(self) -> torch.Tensor:
        """Return the signal-to-noise ratio ``value / stderr``."""
        return self.value / self.stderr

    def __add__(self, other: object) -> GammaMethodEstimate:
        """Return the sample-count-weighted combination of ``self`` and ``other``."""
        if not isinstance(other, GammaMethodEstimate):
            return NotImplemented

        self_batch_shapes = (self.value.shape, self.stderr.shape, self.tau_int.shape)
        other_batch_shapes = (other.value.shape, other.stderr.shape, other.tau_int.shape)
        if len(set(self_batch_shapes + other_batch_shapes)) != 1:
            raise ValueError(
                "GammaMethodEstimate batch shapes must match for addition; "
                f"got self={self_batch_shapes} and other={other_batch_shapes}"
            )

        self_sample_count = self.sample_shapes.sum().to(dtype=self.value.dtype)
        other_sample_count = other.sample_shapes.sum().to(dtype=other.value.dtype)
        total_sample_count = self_sample_count + other_sample_count
        value = (
            self_sample_count * self.value
            + other_sample_count * other.value
        ) / total_sample_count
        stderr = torch.sqrt(
            (self_sample_count * self.stderr).square()
            + (other_sample_count * other.stderr).square()
        ) / total_sample_count
        tau_int = (
            self_sample_count * self.tau_int
            + other_sample_count * other.tau_int
        ) / total_sample_count
        return GammaMethodEstimate(
            value=value,
            stderr=stderr,
            tau_int=tau_int,
            sample_shapes=torch.cat((self.sample_shapes, other.sample_shapes)),
            stderr_of_stderr=None,
            window=None,
            autocovariance=None,
            autocorrelation=None,
        )

    def __radd__(self, other: object) -> GammaMethodEstimate:
        """Return ``self`` for Python ``sum``'s zero start or defer addition."""
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, GammaMethodEstimate):
            return other.__add__(self)
        return NotImplemented


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
        sample_shapes=torch.full(
            (chain_count,),
            sample_count,
            dtype=torch.long,
            device=x.device,
        ),
        stderr_of_stderr=stderr_of_stderr,
        window=window,
        autocovariance=returned_autocovariance,
        autocorrelation=returned_autocorrelation,
    )
