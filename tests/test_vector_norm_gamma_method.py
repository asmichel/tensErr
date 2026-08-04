"""End-to-end integration tests for vector-norm gamma-method estimates."""

from __future__ import annotations

import os

import pytest
import torch

from tensErr import (
    VectorNormGammaMethodEstimate,
    VectorNormGammaMethodHelper,
    gamma_method,
    vector_norm_gamma_method,
)
from tests.ar1 import (
    orthogonal_ar1_mixer,
    symmetric_ar1_model,
    vector_ar1_replica_histories,
    vector_mean_norm_stderr,
)

# NORM_SIGMAS bounds the exact norm residual in reported stderr units.
NORM_SIGMAS = 6.0

# ERR_RTOL bounds finite-window gamma-method stderr variation.
ERR_RTOL = 0.05

# ERR_SIGMAS uses the estimator uncertainty as an integration buffer.
ERR_SIGMAS = 4.0

# LONG_AR1_ENV disables long-autocorrelation tests when set to ``0``.
LONG_AR1_ENV = "TENSERR_LONG_AR1"


def _estimate_norm(
    histories: torch.Tensor,
    *,
    return_autocorrelation: bool,
) -> VectorNormGammaMethodEstimate:
    """Return the helper-driven vector-norm estimate for ``histories``."""
    replica_count, sample_count, _ = histories.shape
    helper = VectorNormGammaMethodHelper(
        replica_count=replica_count,
        sample_count=sample_count,
        accumulation_dtype=torch.float64,
    )
    for replica_index in range(helper.widehat_count):
        helper.accumulate_widehat_term(histories[replica_index].mean(dim=0))
    widehat = helper.finalize_widehat()
    for replica_index in range(helper.widehat_count, replica_count):
        helper.accumulate_widecheck_projection_term(histories[replica_index] @ widehat)
    return helper.compute(
        gamma_method_s=2.0,
        return_autocorrelation=return_autocorrelation,
    )


def _check_ar1_norm(
    *,
    eig_range: tuple[float, float],
    dimension: int,
    m_bar: torch.Tensor,
    mixing: float,
    replica_count: int,
    sample_count: int,
    seed: int,
    return_autocorrelation: bool,
) -> VectorNormGammaMethodEstimate:
    """Assert helper output is calibrated against exact AR(1) norm theory."""
    model = symmetric_ar1_model(
        eig_range,
        dimension,
        m_bar,
        mixing,
    )
    histories = vector_ar1_replica_histories(
        model,
        replica_count,
        sample_count,
        seed=seed,
    )
    estimate = _estimate_norm(
        histories,
        return_autocorrelation=return_autocorrelation,
    )
    exact_norm = torch.linalg.vector_norm(model.mean)
    exact_stderr = vector_mean_norm_stderr(
        model,
        replica_count,
        sample_count,
    )

    assert estimate.replica_count == replica_count
    assert estimate.widehat_count == replica_count // 2
    assert estimate.widecheck_count == replica_count // 2
    torch.testing.assert_close(
        estimate.sample_shapes,
        torch.full((replica_count,), sample_count, dtype=torch.long),
    )
    torch.testing.assert_close(
        estimate.value,
        exact_norm,
        rtol=0.0,
        atol=NORM_SIGMAS * estimate.stderr.item(),
    )
    torch.testing.assert_close(
        estimate.stderr,
        exact_stderr,
        rtol=ERR_RTOL,
        atol=ERR_SIGMAS * estimate.stderr_of_stderr.item(),
    )
    return estimate


def test_ar1_large_even_r() -> None:
    """Check helper estimate against exact non-diagonal symmetric AR(1) theory."""
    sample_count = 16_384
    estimate = _check_ar1_norm(
        eig_range=(0.12, 0.72),
        dimension=5,
        m_bar=torch.tensor([0.85, -0.35, 0.55, 1.10, -0.70], dtype=torch.float64),
        mixing=0.55,
        replica_count=64,
        sample_count=sample_count,
        seed=20260701,
        return_autocorrelation=True,
    )

    assert estimate.autocovariance is not None
    assert estimate.autocorrelation is not None
    assert estimate.Q_bar_autocovariance is not None
    assert estimate.Q_bar_C_f is not None
    assert estimate.autocovariance.shape == torch.Size([sample_count // 2])
    assert estimate.autocorrelation.shape == torch.Size([sample_count // 2])
    assert estimate.Q_bar_autocovariance.shape == torch.Size([sample_count // 2])
    torch.testing.assert_close(
        estimate.Q_bar_C_f,
        2 * estimate.tau_int * estimate.Q_bar_autocovariance[0],
    )
    torch.testing.assert_close(
        estimate.C_f,
        2 * estimate.tau_int * estimate.autocovariance[0],
    )


def test_ar1_mixed_spectrum() -> None:
    """Check helper estimate for a non-diagonal AR(1) with negative eigenvalues."""
    _check_ar1_norm(
        eig_range=(-0.35, 0.55),
        dimension=4,
        m_bar=torch.tensor([1.20, -0.40, 0.65, 0.20], dtype=torch.float64),
        mixing=0.80,
        replica_count=48,
        sample_count=8_192,
        seed=20260702,
        return_autocorrelation=False,
    )


def test_ar1_axis_mean() -> None:
    """Check the exact norm error when ``m_bar`` is the first basis vector."""
    _check_ar1_norm(
        eig_range=(0.08, 0.64),
        dimension=6,
        m_bar=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        mixing=0.65,
        replica_count=64,
        sample_count=8_192,
        seed=20260703,
        return_autocorrelation=False,
    )


@pytest.mark.skipif(
    os.environ.get(LONG_AR1_ENV) == "0",
    reason=f"set {LONG_AR1_ENV}=0 to skip the long-autocorrelation AR(1) test",
)
def test_ar1_long_autocorrelation_projection() -> None:
    """Check that ``m_bar`` projection selects the long-correlation eigendirection."""
    dimension = 6
    eig_range = (0.05, 0.985)
    mixing = 0.70
    mixer = orthogonal_ar1_mixer(dimension, mixing)

    short_estimate = _check_ar1_norm(
        eig_range=eig_range,
        dimension=dimension,
        m_bar=mixer[:, 0],
        mixing=mixing,
        replica_count=64,
        sample_count=131_072,
        seed=20260704,
        return_autocorrelation=False,
    )
    long_estimate = _check_ar1_norm(
        eig_range=eig_range,
        dimension=dimension,
        m_bar=mixer[:, -1],
        mixing=mixing,
        replica_count=64,
        sample_count=131_072,
        seed=20260705,
        return_autocorrelation=False,
    )
    eigenvalues = torch.linspace(
        eig_range[0],
        eig_range[1],
        dimension,
        dtype=torch.float64,
    )
    exact_ratio = (1.0 - eigenvalues[0]) / (1.0 - eigenvalues[-1])

    assert exact_ratio > 50.0
    torch.testing.assert_close(
        long_estimate.stderr / short_estimate.stderr,
        exact_ratio,
        rtol=0.08,
        atol=0.0,
    )


@pytest.mark.skipif(
    os.environ.get(LONG_AR1_ENV) == "0",
    reason=f"set {LONG_AR1_ENV}=0 to skip the long-autocorrelation AR(1) test",
)
def test_ar1_long_autocorrelation_zero_norm() -> None:
    """Check the signed estimate for an exact zero mean with long correlation."""
    dimension = 6
    replica_count = 64
    sample_count = 131_072
    model = symmetric_ar1_model(
        (0.05, 0.985),
        dimension=dimension,
        mean=torch.zeros(dimension, dtype=torch.float64),
        mixing=0.70,
    )
    histories = vector_ar1_replica_histories(
        model,
        replica_count,
        sample_count,
        seed=20260706,
    )
    estimate = _estimate_norm(histories, return_autocorrelation=False)
    exact_norm = torch.linalg.vector_norm(model.mean)

    assert torch.count_nonzero(model.mean) == 0
    assert torch.isfinite(estimate.value)
    assert torch.isfinite(estimate.stderr)
    assert torch.isfinite(estimate.snr)
    assert torch.isfinite(estimate.stderr_of_stderr)
    torch.testing.assert_close(
        estimate.value,
        exact_norm,
        rtol=0.0,
        atol=NORM_SIGMAS * estimate.stderr.item(),
    )


def test_snr_matches_signed_root_delta_method() -> None:
    """Vector norm SNR equals the signed-root ratio with its factor of two."""
    Q_history = torch.tensor(
        [-4.0, -2.0, -5.0, -3.0, -6.0, -1.0, -5.0, -2.0],
        dtype=torch.float64,
    )
    replica_count = 10
    widehat_count = 4
    widecheck_count = replica_count - widehat_count
    Q = gamma_method(Q_history.unsqueeze(-2))
    Q_bar_stderr = 2 * widecheck_count**0.5 / replica_count**0.5 * Q.stderr

    estimate = vector_norm_gamma_method(
        Q_history,
        replica_count=replica_count,
        widehat_count=widehat_count,
    )

    assert estimate.value.item() < 0.0
    assert estimate.stderr.item() > 0.0
    torch.testing.assert_close(estimate.snr, torch.abs(estimate.value) / estimate.stderr)
    torch.testing.assert_close(estimate.snr, torch.abs(2 * Q.value / Q_bar_stderr))


def test_s_zero_reports_Q_replica_mean_variance_without_reconstructing_stderr() -> None:
    """Q_bar_C_f exposes the biased replica-mean variance in the S=0 limit."""
    Q_replica_means = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float64)

    estimate = vector_norm_gamma_method(
        Q_replica_means,
        replica_count=8,
        widehat_count=4,
        gamma_method_s=0.0,
    )

    assert estimate.Q_bar_C_f is not None
    torch.testing.assert_close(
        estimate.Q_bar_C_f,
        torch.var(Q_replica_means, correction=0),
    )
    torch.testing.assert_close(
        estimate.Q_bar_C_f * Q_replica_means.numel() / (Q_replica_means.numel() - 1),
        torch.var(Q_replica_means, correction=1),
    )


def test_addition_preserves_vector_subtype_and_combines_replica_metadata() -> None:
    """Vector addition returns its subtype with sample-weighted common fields."""
    left = vector_norm_gamma_method(
        torch.tensor([1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0]),
        replica_count=4,
        widehat_count=2,
    )
    right = vector_norm_gamma_method(
        torch.tensor([2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0]),
        replica_count=6,
        widehat_count=2,
    )
    left_count = left.sample_shapes.sum().to(dtype=left.value.dtype)
    right_count = right.sample_shapes.sum().to(dtype=right.value.dtype)
    total_count = left_count + right_count

    combined = left + right

    assert isinstance(combined, VectorNormGammaMethodEstimate)
    assert combined.replica_count == 10
    assert combined.widehat_count == 4
    assert combined.widecheck_count == 6
    torch.testing.assert_close(
        combined.sample_shapes,
        torch.tensor([8, 8, 8, 8, 10, 10, 10, 10, 10, 10]),
    )
    torch.testing.assert_close(
        combined.value,
        (left_count * left.value + right_count * right.value) / total_count,
    )
    torch.testing.assert_close(
        combined.stderr,
        torch.sqrt(
            (left_count * left.stderr).square()
            + (right_count * right.stderr).square()
        )
        / total_count,
    )
    torch.testing.assert_close(
        combined.tau_int,
        (left_count * left.tau_int + right_count * right.tau_int) / total_count,
    )
    torch.testing.assert_close(
        combined.C_f,
        (left_count * left.C_f + right_count * right.C_f) / total_count,
    )
    torch.testing.assert_close(combined.snr, torch.abs(combined.value) / combined.stderr)
    assert combined.stderr_of_stderr is None
    assert combined.window is None
    assert combined.autocovariance is None
    assert combined.autocorrelation is None
    assert combined.Q_bar_C_f is None
    assert combined.Q_bar_autocovariance is None


def test_sum_preserves_vector_estimate_subtype() -> None:
    """Python ``sum`` combines vector estimates without erasing their subtype."""
    estimate = vector_norm_gamma_method(
        torch.tensor([1.0, 3.0, 2.0, 4.0, 3.0, 5.0]),
        replica_count=4,
        widehat_count=2,
    )

    combined = sum((estimate, estimate))

    assert isinstance(combined, VectorNormGammaMethodEstimate)
    assert combined.replica_count == 8


def test_addition_rejects_mixed_gamma_estimate_types() -> None:
    """Addition rejects a vector estimate paired with a plain gamma estimate."""
    vector_estimate = vector_norm_gamma_method(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        replica_count=4,
        widehat_count=2,
    )
    scalar_estimate = gamma_method(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(0)
    )

    with pytest.raises(TypeError):
        _ = vector_estimate + scalar_estimate
    with pytest.raises(TypeError):
        _ = scalar_estimate + vector_estimate
