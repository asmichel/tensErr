"""Conditional integration tests for CUDA and HIP Torch accelerators."""

from __future__ import annotations

import pytest
import torch

from tensErr import VectorNormGammaMethodHelper, gamma_method

# pytestmark skips accelerator integration tests when Torch has no CUDA/HIP device.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA or HIP accelerator",
)


def _assert_tensor_fields_match(
    accelerator_result: object,
    cpu_result: object,
    field_names: tuple[str, ...],
) -> None:
    """Compare ``field_names`` tensors and require accelerator result residency."""
    for field_name in field_names:
        accelerator_value = getattr(accelerator_result, field_name)
        cpu_value = getattr(cpu_result, field_name)
        assert isinstance(accelerator_value, torch.Tensor)
        assert isinstance(cpu_value, torch.Tensor)
        assert accelerator_value.device.type == "cuda"
        if cpu_value.is_floating_point():
            torch.testing.assert_close(
                accelerator_value.cpu(),
                cpu_value,
                rtol=1.0e-9,
                atol=1.0e-11,
            )
        else:
            torch.testing.assert_close(accelerator_value.cpu(), cpu_value)


@pytest.mark.parametrize("gamma_method_s", (0.0, 2.0))
def test_gamma_method_matches_cpu_on_accelerator(gamma_method_s: float) -> None:
    """Run gamma_method FFT analysis on CUDA/HIP and compare every tensor field."""
    time = torch.linspace(-3.0, 3.0, 256, dtype=torch.float64)
    samples = torch.stack(
        (
            torch.sin(time) + 0.1 * time,
            torch.cos(1.7 * time) - 0.2 * time,
        )
    )
    cpu_result = gamma_method(
        samples,
        gamma_method_s=gamma_method_s,
        return_autocorrelation=True,
    )
    accelerator_result = gamma_method(
        samples.to("cuda"),
        gamma_method_s=gamma_method_s,
        return_autocorrelation=True,
    )

    _assert_tensor_fields_match(
        accelerator_result,
        cpu_result,
        (
            "value",
            "stderr",
            "tau_int",
            "C_f",
            "sample_shapes",
            "stderr_of_stderr",
            "window",
            "autocovariance",
            "autocorrelation",
        ),
    )


def _vector_helper_result(device: str) -> object:
    """Return a vector-norm helper result accumulated on ``device``."""
    helper = VectorNormGammaMethodHelper(
        replica_count=4,
        sample_count=256,
        widehat_count=2,
    )
    helper.accumulate_widehat_term(
        torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64, device=device)
    )
    helper.accumulate_widehat_term(
        torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64, device=device)
    )
    widehat = helper.finalize_widehat()
    assert widehat.device.type == device

    time = torch.linspace(-2.0, 2.0, 256, dtype=torch.float64, device=device)
    helper.accumulate_widecheck_projection_term(4.0 + torch.sin(time))
    helper.accumulate_widecheck_projection_term(3.0 + torch.cos(1.3 * time))
    return helper.compute(gamma_method_s=2.0, return_autocorrelation=True)


def test_vector_norm_helper_matches_cpu_on_accelerator() -> None:
    """Run stateful vector-norm accumulation on CUDA/HIP and compare with CPU."""
    cpu_result = _vector_helper_result("cpu")
    accelerator_result = _vector_helper_result("cuda")

    _assert_tensor_fields_match(
        accelerator_result,
        cpu_result,
        (
            "value",
            "stderr",
            "stderr_of_stderr",
            "Q_bar_tau_int",
            "Q_bar_C_f",
            "Q_bar_autocovariance",
            "Q_bar_autocorrelation",
        ),
    )
