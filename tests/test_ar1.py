"""Tests for the shared stationary Gaussian AR(1) fixtures."""

import torch

from tests.ar1 import (
    scalar_ar1_replica_histories,
    symmetric_ar1_model,
    vector_ar1_replica_histories,
)


def test_scalar_ar1_is_one_dimensional_vector_specialization() -> None:
    """Scalar and one-dimensional vector fixtures coincide for one seed."""
    rho = 0.65
    mean = 0.4
    replica_count = 7
    sample_count = 23
    seed = 20260804
    scalar_histories = scalar_ar1_replica_histories(
        replica_count,
        sample_count,
        rho=rho,
        mean=mean,
        seed=seed,
    )
    model = symmetric_ar1_model(
        (rho, rho),
        dimension=1,
        mean=torch.tensor([mean], dtype=torch.float64),
        mixing=0.0,
    )
    vector_histories = vector_ar1_replica_histories(
        model,
        replica_count,
        sample_count,
        seed=seed,
    )

    torch.testing.assert_close(vector_histories.squeeze(-1), scalar_histories)
    torch.testing.assert_close(
        model.stationary_covariance.squeeze(),
        torch.tensor(1.0 / (1.0 - rho**2), dtype=torch.float64),
    )
    torch.testing.assert_close(
        model.summed_autocovariance.squeeze(),
        torch.tensor(1.0 / (1.0 - rho) ** 2, dtype=torch.float64),
    )


def test_ar1_fixture_seed_does_not_mutate_global_torch_rng() -> None:
    """A fixture-local ``seed`` leaves Torch's global random stream unchanged."""
    torch.manual_seed(1234)
    expected = torch.randn(8)
    torch.manual_seed(1234)

    scalar_ar1_replica_histories(3, 11, rho=0.5, seed=17)

    torch.testing.assert_close(torch.randn(8), expected)
