"""End-to-end integration tests for vector-norm gamma-method estimates."""

from __future__ import annotations

import os

import pytest
import torch

from torch_uwerr import VectorNormGammaMethodEstimate, VectorNormGammaMethodHelper

# NORM_SIGMAS bounds the exact norm residual in reported stderr units.
NORM_SIGMAS = 6.0

# ERR_RTOL bounds finite-window gamma-method stderr variation.
ERR_RTOL = 0.05

# ERR_SIGMAS uses the estimator uncertainty as an integration buffer.
ERR_SIGMAS = 4.0

# LONG_AR1_ENV disables long-autocorrelation tests when set to ``0``.
LONG_AR1_ENV = "TORCH_UWERR_LONG_AR1"


def _mixer(
    dimension: int,
    mixing: float,
) -> torch.Tensor:
    """Return a deterministic orthogonal mixer for ``dimension`` coordinates."""
    skew = torch.zeros((dimension, dimension), dtype=torch.float64)
    for row in range(dimension):
        for column in range(row + 1, dimension):
            entry = mixing * (-1.0) ** (row + column) / (1.0 + column - row)
            skew[row, column] = entry
            skew[column, row] = -entry
    return torch.linalg.matrix_exp(skew)


def _ar1_model(
    eig_range: tuple[float, float],
    dimension: int,
    m_bar: torch.Tensor,
    mixing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``A``, ``b``, exact ``m_bar``, and ``C`` from the AR(1) theory note."""
    eigenvalues = torch.linspace(
        eig_range[0],
        eig_range[1],
        dimension,
        dtype=torch.float64,
    )
    mixer = _mixer(dimension, mixing)
    A = mixer @ torch.diag(eigenvalues) @ mixer.T
    identity = torch.eye(dimension, dtype=torch.float64)
    b = (identity - A) @ m_bar
    exact_m_bar = torch.linalg.solve(identity - A, b)
    resolvent = torch.linalg.solve(identity - A, identity)
    C = resolvent @ resolvent
    return A, b, exact_m_bar, C


def _ar1_history(
    A: torch.Tensor,
    b: torch.Tensor,
    m_bar: torch.Tensor,
    replica_count: int,
    sample_count: int,
    seed: int,
) -> torch.Tensor:
    """Return deterministic stationary AR(1) histories with unit innovations."""
    dimension = b.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.eye(dimension, dtype=torch.float64)
    stationary_covariance = torch.linalg.solve(identity - A @ A, identity)
    covariance_eigenvalues, covariance_eigenvectors = torch.linalg.eigh(
        stationary_covariance
    )
    stationary_scale = (
        covariance_eigenvectors
        @ torch.diag(torch.sqrt(covariance_eigenvalues))
        @ covariance_eigenvectors.T
    )
    state = (
        m_bar
        + torch.randn(
            (replica_count, dimension),
            generator=generator,
            dtype=torch.float64,
        )
        @ stationary_scale.T
    )
    histories = torch.empty(
        (replica_count, sample_count, dimension),
        dtype=torch.float64,
    )
    histories[:, 0, :] = state
    for sample_index in range(1, sample_count):
        innovations = torch.randn(
            (replica_count, dimension),
            generator=generator,
            dtype=torch.float64,
        )
        state = b + state @ A.T + innovations
        histories[:, sample_index, :] = state
    return histories


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


def _norm_error(
    m_bar: torch.Tensor,
    C: torch.Tensor,
    replica_count: int,
    sample_count: int,
) -> torch.Tensor:
    """Return exact AR(1) norm stderr from ``m_bar`` and summed autocovariance ``C``."""
    norm = torch.linalg.vector_norm(m_bar)
    norm_variance = (m_bar @ C @ m_bar) / (replica_count * sample_count * norm * norm)
    return torch.sqrt(norm_variance)


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
    A, b, exact_m_bar, C = _ar1_model(
        eig_range,
        dimension,
        m_bar,
        mixing,
    )
    histories = _ar1_history(
        A,
        b,
        exact_m_bar,
        replica_count,
        sample_count,
        seed,
    )
    estimate = _estimate_norm(
        histories,
        return_autocorrelation=return_autocorrelation,
    )
    exact_norm = torch.linalg.vector_norm(exact_m_bar)
    exact_stderr = _norm_error(
        exact_m_bar,
        C,
        replica_count,
        sample_count,
    )

    assert estimate.replica_count == replica_count
    assert estimate.widehat_count == replica_count // 2
    assert estimate.widecheck_count == replica_count // 2
    assert estimate.sample_count == sample_count
    torch.testing.assert_close(
        estimate.norm,
        exact_norm,
        rtol=0.0,
        atol=NORM_SIGMAS * estimate.norm_stderr.item(),
    )
    torch.testing.assert_close(
        estimate.norm_stderr,
        exact_stderr,
        rtol=ERR_RTOL,
        atol=ERR_SIGMAS * estimate.norm_stderr_of_stderr.item(),
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

    assert estimate.Q_bar_autocovariance is not None
    assert estimate.Q_bar_autocorrelation is not None
    assert estimate.Q_bar_autocovariance.shape == torch.Size([sample_count // 2])
    assert estimate.Q_bar_autocorrelation.shape == torch.Size([sample_count // 2])


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
    mixer = _mixer(dimension, mixing)

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
        long_estimate.norm_stderr / short_estimate.norm_stderr,
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
    A, b, exact_m_bar, _ = _ar1_model(
        eig_range=(0.05, 0.985),
        dimension=dimension,
        m_bar=torch.zeros(dimension, dtype=torch.float64),
        mixing=0.70,
    )
    histories = _ar1_history(
        A,
        b,
        exact_m_bar,
        replica_count,
        sample_count,
        seed=20260706,
    )
    estimate = _estimate_norm(histories, return_autocorrelation=False)
    exact_norm = torch.linalg.vector_norm(exact_m_bar)

    assert torch.count_nonzero(exact_m_bar) == 0
    assert torch.isfinite(estimate.norm)
    assert torch.isfinite(estimate.norm_stderr)
    assert torch.isfinite(estimate.norm_snr)
    assert torch.isfinite(estimate.norm_stderr_of_stderr)
    torch.testing.assert_close(
        estimate.norm,
        exact_norm,
        rtol=0.0,
        atol=NORM_SIGMAS * estimate.norm_stderr.item(),
    )
