"""Stationary Gaussian AR(1) fixtures shared across the test suite.

The scalar and symmetric vector processes implement the conventions recorded in
``tests/ar1_model_conventions.md``: unit innovation covariance, exact stationary
initialization, and seeded CPU generation that does not mutate Torch's global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SymmetricAR1Model:
    """Exact matrices for a stable symmetric vector AR(1) process.

    ``transition`` is the symmetric matrix ``A``; ``offset`` is ``b``;
    ``mean`` is ``(I - A)^{-1} b``; ``stationary_covariance`` is
    ``(I - A^2)^{-1}``; and ``summed_autocovariance`` is ``(I - A)^{-2}``.
    """

    transition: torch.Tensor
    offset: torch.Tensor
    mean: torch.Tensor
    stationary_covariance: torch.Tensor
    summed_autocovariance: torch.Tensor


def scalar_ar1_histories(
    history_shape: tuple[int, ...],
    *,
    rho: float,
    seed: int,
    mean: float = 0.0,
) -> torch.Tensor:
    """Return stationary scalar AR(1) histories shaped as ``history_shape``.

    The final axis is time. Each independent leading-index process obeys
    ``x_t = mean + rho * (x_{t-1} - mean) + epsilon_t`` with unit-variance
    Gaussian ``epsilon_t`` and stationary initial variance ``1 / (1 - rho^2)``.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    leading_shape = history_shape[:-1]
    sample_count = history_shape[-1]
    stationary_scale = (1.0 - rho**2) ** -0.5
    state = mean + stationary_scale * torch.randn(
        leading_shape,
        generator=generator,
        dtype=torch.float64,
    )
    histories = torch.empty(history_shape, dtype=torch.float64)
    histories[..., 0] = state
    for sample_index in range(1, sample_count):
        innovation = torch.randn(
            leading_shape,
            generator=generator,
            dtype=torch.float64,
        )
        state = mean + rho * (state - mean) + innovation
        histories[..., sample_index] = state
    return histories


def scalar_ar1_replica_histories(
    replica_count: int,
    sample_count: int,
    *,
    rho: float,
    seed: int,
    mean: float = 0.0,
) -> torch.Tensor:
    """Return ``replica_count`` stationary scalar histories of length ``sample_count``."""
    return scalar_ar1_histories(
        (replica_count, sample_count),
        rho=rho,
        seed=seed,
        mean=mean,
    )


def batched_scalar_ar1_replica_histories(
    batch_count: int,
    replica_count: int,
    sample_count: int,
    *,
    rho: float,
    seed: int,
    mean: float = 0.0,
) -> torch.Tensor:
    """Return ``batch_count`` batches of stationary scalar replica histories."""
    return scalar_ar1_histories(
        (batch_count, replica_count, sample_count),
        rho=rho,
        seed=seed,
        mean=mean,
    )


def orthogonal_ar1_mixer(dimension: int, mixing: float) -> torch.Tensor:
    """Return the deterministic orthogonal eigenvector mixer for vector fixtures."""
    skew = torch.zeros((dimension, dimension), dtype=torch.float64)
    for row in range(dimension):
        for column in range(row + 1, dimension):
            entry = mixing * (-1.0) ** (row + column) / (1.0 + column - row)
            skew[row, column] = entry
            skew[column, row] = -entry
    return torch.linalg.matrix_exp(skew)


def symmetric_ar1_model(
    eigenvalue_range: tuple[float, float],
    dimension: int,
    mean: torch.Tensor,
    mixing: float,
) -> SymmetricAR1Model:
    """Return exact matrices for the symmetric vector convention and parameters."""
    eigenvalues = torch.linspace(
        eigenvalue_range[0],
        eigenvalue_range[1],
        dimension,
        dtype=torch.float64,
    )
    mixer = orthogonal_ar1_mixer(dimension, mixing)
    transition = mixer @ torch.diag(eigenvalues) @ mixer.T
    identity = torch.eye(dimension, dtype=torch.float64)
    offset = (identity - transition) @ mean
    exact_mean = torch.linalg.solve(identity - transition, offset)
    stationary_covariance = torch.linalg.solve(
        identity - transition @ transition,
        identity,
    )
    resolvent = torch.linalg.solve(identity - transition, identity)
    return SymmetricAR1Model(
        transition=transition,
        offset=offset,
        mean=exact_mean,
        stationary_covariance=stationary_covariance,
        summed_autocovariance=resolvent @ resolvent,
    )


def vector_ar1_replica_histories(
    model: SymmetricAR1Model,
    replica_count: int,
    sample_count: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Return stationary vector histories for ``model`` with unit innovations."""
    dimension = model.offset.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    covariance_eigenvalues, covariance_eigenvectors = torch.linalg.eigh(
        model.stationary_covariance
    )
    stationary_scale = (
        covariance_eigenvectors
        @ torch.diag(torch.sqrt(covariance_eigenvalues))
        @ covariance_eigenvectors.T
    )
    state = (
        model.mean
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
        state = model.offset + state @ model.transition.T + innovations
        histories[:, sample_index, :] = state
    return histories


def vector_mean_norm_stderr(
    model: SymmetricAR1Model,
    replica_count: int,
    sample_count: int,
) -> torch.Tensor:
    """Return the exact asymptotic standard error of ``model.mean``'s norm."""
    norm = torch.linalg.vector_norm(model.mean)
    norm_variance = (
        model.mean @ model.summed_autocovariance @ model.mean
    ) / (replica_count * sample_count * norm * norm)
    return torch.sqrt(norm_variance)
