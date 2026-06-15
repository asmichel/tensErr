"""Tests for the public torch_uwerr gamma-method API."""

import torch

from torch_uwerr import GammaMethodEstimate, gamma_method_mean


def direct_autocovariance(samples: torch.Tensor) -> torch.Tensor:
    """Return direct lag-loop autocovariance for dense chain samples."""
    x = samples.to(dtype=torch.float64)
    if x.ndim == 1:
        x = x.reshape(1, x.shape[0])
    centered = x - x.mean(dim=-1, keepdim=True)
    sample_count = x.shape[-1]
    chain_count = x.shape[-2]
    γ = []
    for lag in range(sample_count // 2):
        pairs = centered[..., :, : sample_count - lag] * centered[..., :, lag:]
        γ.append(pairs.sum(dim=(-2, -1)) / (chain_count * (sample_count - lag)))
    return torch.stack(γ, dim=-1)


def test_shape_behavior_for_unbatched_and_batched_histories() -> None:
    """gamma_method_mean returns scalar or batch-shaped tensors by input rank."""
    one_chain = torch.arange(7, dtype=torch.float32)
    two_chains = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    batched = torch.arange(70, dtype=torch.float32).reshape(5, 2, 7)

    one_estimate = gamma_method_mean(one_chain, return_autocorrelation=True)
    two_estimate = gamma_method_mean(two_chains, return_autocorrelation=True)
    batched_estimate = gamma_method_mean(batched, return_autocorrelation=True)

    assert isinstance(one_estimate, GammaMethodEstimate)
    assert one_estimate.value.shape == torch.Size([])
    assert two_estimate.value.shape == torch.Size([])
    assert batched_estimate.value.shape == torch.Size([5])
    assert one_estimate.autocovariance is not None
    assert two_estimate.autocovariance is not None
    assert batched_estimate.autocovariance is not None
    assert one_estimate.autocovariance.shape == torch.Size([3])
    assert two_estimate.autocovariance.shape == torch.Size([3])
    assert batched_estimate.autocovariance.shape == torch.Size([5, 3])


def test_user_moved_data_normalizes_to_batch_chain_sample_layout() -> None:
    """User-moved axes work after normalizing into ``(*B, R, N)`` layout."""
    raw = torch.arange(2 * 7 * 3, dtype=torch.float64).reshape(2, 7, 3)
    normalized = torch.movedim(raw, -1, -2)

    estimate = gamma_method_mean(normalized)
    reference = gamma_method_mean(normalized.reshape(2, 3, 7))

    assert estimate.value.shape == torch.Size([2])
    torch.testing.assert_close(estimate.value, reference.value)
    torch.testing.assert_close(estimate.stderr, reference.stderr)


def test_fft_autocovariance_matches_direct_lag_loop() -> None:
    """FFT autocovariance agrees with the dense torch lag-loop reference."""
    samples = torch.tensor(
        [
            [[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 8.0, 7.0]],
            [[2.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0]],
        ],
        dtype=torch.float64,
    )

    estimate = gamma_method_mean(samples, return_autocorrelation=True)

    assert estimate.autocovariance is not None
    torch.testing.assert_close(estimate.autocovariance, direct_autocovariance(samples))


def test_zero_variance_uses_pyerrors_degenerate_values() -> None:
    """A constant dense history has zero stderr and tau_int of one half."""
    samples = torch.full((3, 9), 4.25)

    estimate = gamma_method_mean(samples, return_autocorrelation=True)

    assert estimate.autocovariance is not None
    assert estimate.autocorrelation is not None
    torch.testing.assert_close(estimate.value, torch.tensor(4.25, dtype=torch.float64))
    torch.testing.assert_close(estimate.stderr, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(estimate.tau_int, torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(estimate.stderr_of_stderr, torch.tensor(0.0, dtype=torch.float64))
    assert estimate.window.item() == 0
    torch.testing.assert_close(estimate.autocovariance, torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(estimate.autocorrelation, torch.zeros(4, dtype=torch.float64))


def test_s_zero_standard_error_branch() -> None:
    """S=0 disables autocorrelation analysis and uses pyerrors standard error."""
    samples = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]])
    γ = direct_autocovariance(samples)
    total_count = samples.numel()

    estimate = gamma_method_mean(samples, gamma_method_s=0.0)

    torch.testing.assert_close(estimate.tau_int, torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(estimate.stderr, torch.sqrt(γ[0] / (total_count - 1)))
    torch.testing.assert_close(
        estimate.stderr_of_stderr,
        estimate.stderr * torch.sqrt(torch.tensor(0.5 / total_count, dtype=torch.float64)),
    )
    assert estimate.window.item() == 0
    assert estimate.autocovariance is None
    assert estimate.autocorrelation is None


def test_s_two_automatic_window_branch() -> None:
    """S=2.0 follows pyerrors automatic windowing and bias-corrected stderr."""
    samples = torch.tensor([[1.0, 1.5, 2.5, 3.0, 5.0, 8.0, 13.0, 21.0]])
    γ = direct_autocovariance(samples)
    ρ = γ / γ[..., :1]
    total_count = samples.numel()
    τ_history = torch.cumsum(torch.cat((torch.tensor([0.5], dtype=torch.float64), ρ[1:])), dim=-1)
    τ_history = torch.maximum(τ_history, torch.full_like(τ_history, 0.5 + torch.finfo(torch.float64).eps))
    τ_tail = 2.0 / torch.log((2 * τ_history[1:] + 1) / (2 * τ_history[1:] - 1))
    candidates = torch.arange(1, γ.shape[-1], dtype=torch.float64)
    g_w = torch.exp(-candidates / τ_tail) - τ_tail / torch.sqrt(candidates * total_count)
    conditions = g_w < 0
    conditions[-1] = True
    window = int(conditions.to(torch.long).argmax().item() + 1)
    tau_int = τ_history[window] * (1 + (2 * window + 1) / total_count) / (1 + 1 / total_count)
    stderr = torch.sqrt(2 * tau_int * γ[0] * (1 + 1 / total_count) / total_count)

    estimate = gamma_method_mean(samples, gamma_method_s=2.0)

    assert estimate.window.item() == window
    torch.testing.assert_close(estimate.tau_int, tau_int)
    torch.testing.assert_close(estimate.stderr, stderr)
    torch.testing.assert_close(
        estimate.stderr_of_stderr,
        stderr * torch.sqrt(torch.tensor((window + 0.5) / total_count, dtype=torch.float64)),
    )
