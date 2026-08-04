"""Tests for the public shared estimate interface."""

from dataclasses import fields

import torch

from tensErr import (
    Estimate,
    GammaMethodEstimate,
    VectorNormGammaMethodEstimate,
    gamma_method,
    vector_norm_gamma_method,
)


def test_estimate_base_contains_only_universal_fields() -> None:
    """Estimate declares only value and stderr as shared stored fields."""
    assert tuple(field.name for field in fields(Estimate)) == ("value", "stderr")


def test_snr_uses_absolute_ratio_and_infinite_zero_error_contract() -> None:
    """Estimate.snr handles signs and both zero-error value cases elementwise."""
    estimate = Estimate(
        value=torch.tensor([-3.0, 0.0, 0.0, 4.0], dtype=torch.float64),
        stderr=torch.tensor([1.5, 2.0, 0.0, 0.0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        estimate.snr,
        torch.tensor([2.0, 0.0, torch.inf, torch.inf], dtype=torch.float64),
    )


def test_gamma_and_vector_results_inherit_estimate() -> None:
    """Both public result types implement the shared Estimate interface."""
    gamma_estimate = gamma_method(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    )
    vector_estimate = vector_norm_gamma_method(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        replica_count=4,
        widehat_count=2,
    )

    assert isinstance(gamma_estimate, Estimate)
    assert isinstance(gamma_estimate, GammaMethodEstimate)
    assert isinstance(vector_estimate, Estimate)
    assert isinstance(vector_estimate, VectorNormGammaMethodEstimate)
    assert gamma_estimate.stderr_of_stderr is not None
    assert isinstance(vector_estimate.stderr_of_stderr, torch.Tensor)


def test_vector_result_omits_version_one_estimate_aliases() -> None:
    """Vector results expose only the shared version 0.2 estimate field names."""
    estimate = vector_norm_gamma_method(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        replica_count=4,
        widehat_count=2,
    )

    assert not hasattr(estimate, "norm")
    assert not hasattr(estimate, "norm_stderr")
    assert not hasattr(estimate, "norm_snr")
    assert not hasattr(estimate, "norm_stderr_of_stderr")
