"""Tests for the public torch_uwerr gamma-method API."""

import pytest
import torch

from torch_uwerr import (
    GammaMethodEstimate,
    GammaMethodMeanAccumulator,
    gamma_method_mean,
)


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


def manual_dense_chains(
    chunk: torch.Tensor,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``chunk`` normalized with gamma_method_mean's 1D replica semantics."""
    x = torch.as_tensor(chunk).to(dtype=accumulation_dtype)
    if x.ndim == 1:
        return x.reshape(1, x.shape[0])
    return x


def _ar1_history(
    history_shape: tuple[int, ...],
    *,
    rho: float,
    seed: int,
) -> torch.Tensor:
    """Return an AR(1) tensor shaped as ``history_shape`` using ``rho`` and ``seed``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    leading_shape = history_shape[:-1]
    sample_count = history_shape[-1]
    history = torch.empty(history_shape, dtype=torch.float64)
    history[..., 0] = torch.randn(
        leading_shape,
        generator=generator,
        dtype=torch.float64,
    )
    eps = torch.randn(
        (*leading_shape, sample_count - 1),
        generator=generator,
        dtype=torch.float64,
    )
    innovation_scale = (1.0 - rho**2) ** 0.5
    for sample_index in range(1, sample_count):
        history[..., sample_index] = (
            rho * history[..., sample_index - 1]
            + innovation_scale * eps[..., sample_index - 1]
        )
    return history


def ar1_replica_history(
    replica_count: int,
    sample_count: int,
    *,
    rho: float,
    seed: int,
) -> torch.Tensor:
    """Return ``replica_count`` seeded AR(1) histories with ``sample_count`` samples."""
    return _ar1_history((replica_count, sample_count), rho=rho, seed=seed)


def ar1_batched_replica_history(
    batch_count: int,
    replica_count: int,
    sample_count: int,
    *,
    rho: float,
    seed: int,
) -> torch.Tensor:
    """Return ``batch_count`` batches of seeded AR(1) replica histories."""
    return _ar1_history((batch_count, replica_count, sample_count), rho=rho, seed=seed)


def manual_chunked_gamma_method_mean(
    chunks: tuple[torch.Tensor, ...],
    *,
    gamma_method_s: float = 2.0,
    accumulation_dtype: torch.dtype = torch.float64,
) -> GammaMethodEstimate:
    """Return the manual replica-weighted combination of chunk estimates."""
    dense_chunks = [manual_dense_chains(chunk, accumulation_dtype) for chunk in chunks]
    batch_shape = dense_chunks[0].shape[:-2]
    replica_count = sum(chunk.shape[-2] for chunk in dense_chunks)
    dtype = dense_chunks[0].dtype
    device = dense_chunks[0].device
    value_numerator = torch.zeros(batch_shape, dtype=dtype, device=device)
    stderr_square_numerator = torch.zeros(batch_shape, dtype=dtype, device=device)
    tau_int_numerator = torch.zeros(batch_shape, dtype=dtype, device=device)

    for chunk in dense_chunks:
        chunk_replica_count = chunk.shape[-2]
        chunk_estimate = gamma_method_mean(
            chunk,
            gamma_method_s=gamma_method_s,
            accumulation_dtype=accumulation_dtype,
        )
        value_numerator = value_numerator + chunk_replica_count * chunk_estimate.value
        weighted_stderr = chunk_replica_count * chunk_estimate.stderr
        stderr_square_numerator = stderr_square_numerator + weighted_stderr.square()
        tau_int_numerator = (
            tau_int_numerator + chunk_replica_count * chunk_estimate.tau_int
        )

    return GammaMethodEstimate(
        value=value_numerator / replica_count,
        stderr=torch.sqrt(stderr_square_numerator) / replica_count,
        tau_int=tau_int_numerator / replica_count,
        stderr_of_stderr=None,
        window=None,
        autocovariance=None,
        autocorrelation=None,
    )


def accumulator_estimate(
    chunks: tuple[torch.Tensor, ...],
    *,
    gamma_method_s: float = 2.0,
) -> GammaMethodEstimate:
    """Return the planned GammaMethodMeanAccumulator estimate for ``chunks``."""
    accumulator = GammaMethodMeanAccumulator(
        gamma_method_s=gamma_method_s,
        accumulation_dtype=torch.float64,
    )
    for chunk in chunks:
        accumulator.accumulate(chunk)
    return accumulator.result()


def assert_accumulator_diagnostics_absent(estimate: GammaMethodEstimate) -> None:
    """Assert accumulator ``estimate`` omits diagnostics requiring dense history."""
    assert estimate.stderr_of_stderr is None
    assert estimate.window is None
    assert estimate.autocovariance is None
    assert estimate.autocorrelation is None


def assert_estimates_close(
    actual: GammaMethodEstimate,
    expected: GammaMethodEstimate,
) -> None:
    """Assert that every public ``GammaMethodEstimate`` field matches."""
    torch.testing.assert_close(actual.value, expected.value)
    torch.testing.assert_close(actual.stderr, expected.stderr)
    torch.testing.assert_close(actual.tau_int, expected.tau_int)
    if expected.stderr_of_stderr is None:
        assert actual.stderr_of_stderr is None
    else:
        assert actual.stderr_of_stderr is not None
        torch.testing.assert_close(actual.stderr_of_stderr, expected.stderr_of_stderr)
    if expected.window is None:
        assert actual.window is None
    else:
        assert actual.window is not None
        torch.testing.assert_close(actual.window, expected.window)
    if expected.autocovariance is None:
        assert actual.autocovariance is None
    else:
        assert actual.autocovariance is not None
        torch.testing.assert_close(actual.autocovariance, expected.autocovariance)
    if expected.autocorrelation is None:
        assert actual.autocorrelation is None
    else:
        assert actual.autocorrelation is not None
        torch.testing.assert_close(actual.autocorrelation, expected.autocorrelation)


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


def test_accumulator_matches_manual_combination_for_uneven_chunks() -> None:
    """GammaMethodMeanAccumulator combines uneven replica chunks exactly."""
    history = ar1_replica_history(
        replica_count=6,
        sample_count=14,
        rho=0.65,
        seed=1729,
    )
    chunks = torch.split(history, (2, 1, 3), dim=-2)

    actual = accumulator_estimate(chunks)
    expected = manual_chunked_gamma_method_mean(chunks)

    assert_accumulator_diagnostics_absent(actual)
    assert_estimates_close(actual, expected)


def test_accumulator_matches_manual_combination_for_one_replica_chunks() -> None:
    """GammaMethodMeanAccumulator accepts chunks containing one replica each."""
    history = ar1_replica_history(
        replica_count=4,
        sample_count=12,
        rho=0.7,
        seed=2718,
    )
    chunks = tuple(history[index : index + 1] for index in range(history.shape[-2]))

    actual = accumulator_estimate(chunks)
    expected = manual_chunked_gamma_method_mean(chunks)

    assert_accumulator_diagnostics_absent(actual)
    assert_estimates_close(actual, expected)


def test_accumulator_matches_manual_combination_for_1d_replica_chunks() -> None:
    """GammaMethodMeanAccumulator treats each 1D chunk as one complete replica."""
    history = ar1_replica_history(
        replica_count=4,
        sample_count=12,
        rho=0.7,
        seed=31415,
    )
    chunks = tuple(history[index] for index in range(history.shape[-2]))

    actual = accumulator_estimate(chunks)
    expected = manual_chunked_gamma_method_mean(chunks)

    assert actual.value.shape == torch.Size([])
    assert_accumulator_diagnostics_absent(actual)
    assert_estimates_close(actual, expected)


def test_accumulator_matches_manual_combination_for_batched_histories() -> None:
    """GammaMethodMeanAccumulator preserves leading batch dimensions."""
    history = ar1_batched_replica_history(
        batch_count=3,
        replica_count=5,
        sample_count=16,
        rho=0.6,
        seed=1618,
    )
    chunks = torch.split(history, (2, 1, 2), dim=-2)

    actual = accumulator_estimate(chunks)
    expected = manual_chunked_gamma_method_mean(chunks)

    assert actual.value.shape == torch.Size([3])
    assert_accumulator_diagnostics_absent(actual)
    assert_estimates_close(actual, expected)


def test_accumulator_matches_manual_combination_for_s_zero() -> None:
    """GammaMethodMeanAccumulator handles the ``S=0`` standard-error branch."""
    history = ar1_replica_history(
        replica_count=5,
        sample_count=10,
        rho=0.5,
        seed=57721,
    )
    chunks = torch.split(history, (3, 2), dim=-2)

    actual = accumulator_estimate(chunks, gamma_method_s=0.0)
    expected = manual_chunked_gamma_method_mean(chunks, gamma_method_s=0.0)

    assert_accumulator_diagnostics_absent(actual)
    assert_estimates_close(actual, expected)


def test_accumulator_matches_manual_combination_for_zero_variance() -> None:
    """GammaMethodMeanAccumulator keeps zero-variance histories finite."""
    chunks = (
        torch.full((2, 10), 4.25, dtype=torch.float64),
        torch.full((1, 10), 4.25, dtype=torch.float64),
    )

    actual = accumulator_estimate(chunks)
    expected = manual_chunked_gamma_method_mean(chunks)

    assert_accumulator_diagnostics_absent(actual)
    assert_estimates_close(actual, expected)


def test_accumulator_result_before_any_chunks_raises_runtime_error() -> None:
    """GammaMethodMeanAccumulator.result raises before any chunks are accumulated."""
    accumulator = GammaMethodMeanAccumulator(accumulation_dtype=torch.float64)

    with pytest.raises(RuntimeError, match="accumulate"):
        accumulator.result()


def test_accumulator_rejects_incompatible_shape_after_first_chunk() -> None:
    """GammaMethodMeanAccumulator rejects later chunks with incompatible shapes."""
    accumulator = GammaMethodMeanAccumulator(accumulation_dtype=torch.float64)
    accumulator.accumulate(torch.zeros((2, 4), dtype=torch.float64))

    with pytest.raises(ValueError, match="batch shape and sample_count"):
        accumulator.accumulate(torch.zeros((2, 5), dtype=torch.float64))
    with pytest.raises(ValueError, match="batch shape and sample_count"):
        accumulator.accumulate(torch.zeros((1, 2, 4), dtype=torch.float64))


def test_accumulator_ar1_fixture_value_matches_and_uncertainty_diverges() -> None:
    """GammaMethodMeanAccumulator matches seeded AR(1) value but not pooled diagnostics."""
    history = ar1_replica_history(
        replica_count=6,
        sample_count=48,
        rho=0.82,
        seed=90210,
    )
    chunks = torch.split(history, (2, 1, 3), dim=-2)

    actual = accumulator_estimate(chunks)
    expected = gamma_method_mean(history)
    stderr_divergence = (actual.stderr - expected.stderr).abs()
    tau_int_divergence = (actual.tau_int - expected.tau_int).abs()

    assert_accumulator_diagnostics_absent(actual)
    torch.testing.assert_close(actual.value, expected.value)
    assert torch.isfinite(stderr_divergence).item()
    assert torch.isfinite(tau_int_divergence).item()
    assert stderr_divergence.item() >= 0.0
    assert tau_int_divergence.item() >= 0.0
    torch.testing.assert_close(
        stderr_divergence,
        torch.tensor(0.02644710374078839, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual.stderr / expected.stderr,
        torch.tensor(1.2187913728302469, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        tau_int_divergence,
        torch.tensor(0.9856817271764671, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual.tau_int / expected.tau_int,
        torch.tensor(1.4050483640558546, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_accumulator_ar1_ratios_move_toward_dense_with_more_replicas() -> None:
    """More seeded AR(1) replicas move chunked ratios closer to dense estimates."""
    replica_counts = (8, 16, 32, 64)
    full_history = ar1_replica_history(
        replica_count=max(replica_counts),
        sample_count=24,
        rho=0.5,
        seed=1234,
    )
    stderr_ratios = []
    tau_int_ratios = []

    for replica_count in replica_counts:
        history = full_history[:replica_count]
        chunk_sizes = (replica_count // 4,) * 4
        chunks = torch.split(history, chunk_sizes, dim=-2)

        actual = accumulator_estimate(chunks)
        expected = gamma_method_mean(history)

        assert_accumulator_diagnostics_absent(actual)
        torch.testing.assert_close(actual.value, expected.value)
        assert torch.isfinite(actual.stderr).item()
        assert torch.isfinite(actual.tau_int).item()
        assert torch.isfinite(expected.stderr).item()
        assert torch.isfinite(expected.tau_int).item()
        assert actual.stderr.item() > 0.0
        assert actual.tau_int.item() > 0.0
        assert expected.stderr.item() > 0.0
        assert expected.tau_int.item() > 0.0
        stderr_ratios.append(actual.stderr / expected.stderr)
        tau_int_ratios.append(actual.tau_int / expected.tau_int)

    stderr_ratio = torch.stack(stderr_ratios)
    tau_int_ratio = torch.stack(tau_int_ratios)
    stderr_distance = (stderr_ratio - 1.0).abs()
    tau_int_distance = (tau_int_ratio - 1.0).abs()

    torch.testing.assert_close(
        stderr_ratio,
        torch.tensor(
            [
                1.246739593582284,
                1.163023571870329,
                1.0702056531638289,
                1.070701989881707,
            ],
            dtype=torch.float64,
        ),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        tau_int_ratio,
        torch.tensor(
            [
                1.582533400277406,
                1.3403826954786817,
                1.122178943122221,
                1.1413860839166483,
            ],
            dtype=torch.float64,
        ),
        rtol=1e-10,
        atol=1e-12,
    )
    assert stderr_distance[-1].item() < stderr_distance[0].item()
    assert tau_int_distance[-1].item() < tau_int_distance[0].item()
    assert stderr_distance[-1].item() < 0.08
    assert tau_int_distance[-1].item() < 0.15
