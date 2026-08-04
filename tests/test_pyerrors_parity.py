"""Parity tests against pyerrors for gamma_method."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from tests.ar1 import scalar_ar1_replica_histories

# FIELD_ALIASES maps tensErr result field names to accepted API spellings.
FIELD_ALIASES = {
    "value": ("value", "mean"),
    "stderr": ("stderr", "dvalue", "error", "err"),
    "window": ("window", "windowsize", "window_size"),
    "tau_int": ("tau_int", "tauint", "tau"),
}

# RESULT_TUPLE_ORDER is the expected positional return order for tuple APIs.
RESULT_TUPLE_ORDER = ("value", "stderr", "window", "tau_int")

# PYERRORS_ENSEMBLE_NAME is the pyerrors ensemble label for replica grouping.
PYERRORS_ENSEMBLE_NAME = "ens"

# FIXTURE_SAMPLE_COUNT fixes deterministic fixture histories at a nontrivial length.
FIXTURE_SAMPLE_COUNT = 96

# AR1_SAMPLE_COUNT keeps AR(1) parity tests stable without making pyerrors slow.
AR1_SAMPLE_COUNT = 4096

# AR1_REPLICA_COUNT gives each rho case several independent histories.
AR1_REPLICA_COUNT = 4

# AR1_RHOS are the requested Gaussian AR(1) autocorrelation coefficients.
AR1_RHOS = (0.0, 0.5, 0.9)

# PARITY_RTOL allows small FFT and tensor-reduction roundoff differences.
PARITY_RTOL = 5.0e-8

# PARITY_ATOL allows near-zero means and errors to compare robustly.
PARITY_ATOL = 5.0e-10


def _pyerrors_one_reference(histories: np.ndarray) -> dict[str, float | int]:
    """Return pyerrors fields for one dense chain set shaped ``(R, N)``."""
    pyerrors = pytest.importorskip("pyerrors")
    samples = [np.asarray(row, dtype=np.float64) for row in histories]
    names = [f"{PYERRORS_ENSEMBLE_NAME}|r{i}" for i in range(histories.shape[0])]
    obs = pyerrors.Obs(samples, names)
    obs.gamma_method(fft=True)
    return {
        "value": float(obs.value),
        "stderr": float(obs.dvalue),
        "window": int(obs.e_windowsize[PYERRORS_ENSEMBLE_NAME]),
        "tau_int": float(obs.e_tauint[PYERRORS_ENSEMBLE_NAME]),
    }


def _pyerrors_reference(histories: np.ndarray) -> dict[str, np.ndarray]:
    """Return pyerrors fields with public ``(..., R, N)`` chain semantics."""
    x = np.asarray(histories, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, x.shape[0])
    batch_shape = x.shape[:-2]
    flattened = x.reshape((-1,) + x.shape[-2:]) if batch_shape else x.reshape((1,) + x.shape)
    references = [_pyerrors_one_reference(chains) for chains in flattened]
    return {
        "value": np.asarray([row["value"] for row in references], dtype=np.float64).reshape(batch_shape),
        "stderr": np.asarray([row["stderr"] for row in references], dtype=np.float64).reshape(batch_shape),
        "window": np.asarray([row["window"] for row in references], dtype=np.int64).reshape(batch_shape),
        "tau_int": np.asarray([row["tau_int"] for row in references], dtype=np.float64).reshape(batch_shape),
    }


def _gamma_method(histories: np.ndarray) -> object:
    """Call tensErr.gamma_method on CPU float64 histories."""
    torch = pytest.importorskip("torch")
    tenserr = pytest.importorskip("tensErr")
    tensor = torch.as_tensor(histories, dtype=torch.float64, device="cpu")
    return tenserr.gamma_method(tensor)


def _as_numpy_array(value: object) -> np.ndarray:
    """Convert tensor-like gamma_method field values to NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _field_from_mapping(result: Mapping[object, object], field: str) -> object:
    """Return field from a mapping result using accepted alias names."""
    for alias in FIELD_ALIASES[field]:
        if alias in result:
            return result[alias]
    raise AssertionError(f"gamma_method result does not contain {field!r}")


def _field_from_attributes(result: object, field: str) -> object:
    """Return field from an object result using accepted alias names."""
    for alias in FIELD_ALIASES[field]:
        if hasattr(result, alias):
            return getattr(result, alias)
    raise AssertionError(f"gamma_method result does not expose {field!r}")


def _normalize_result(result: object) -> dict[str, np.ndarray]:
    """Normalize gamma_method output to value, stderr, window, and tau_int arrays."""
    if isinstance(result, Mapping):
        return {
            field: _as_numpy_array(_field_from_mapping(result, field))
            for field in RESULT_TUPLE_ORDER
        }
    if isinstance(result, tuple):
        if hasattr(result, "_fields"):
            return {
                field: _as_numpy_array(_field_from_attributes(result, field))
                for field in RESULT_TUPLE_ORDER
            }
        assert len(result) == len(RESULT_TUPLE_ORDER)
        return {
            field: _as_numpy_array(value)
            for field, value in zip(RESULT_TUPLE_ORDER, result, strict=True)
        }
    return {
        field: _as_numpy_array(_field_from_attributes(result, field))
        for field in RESULT_TUPLE_ORDER
    }


def _fixed_histories() -> np.ndarray:
    """Return deterministic dense fixtures with positive and negative autocorrelation."""
    t = np.linspace(-1.0, 1.0, FIXTURE_SAMPLE_COUNT, dtype=np.float64)
    return np.stack(
        [
            0.75 * t + np.sin(5.0 * t),
            np.cos(4.0 * np.pi * t) + 0.15 * t,
            (t * t - np.mean(t * t)) + 0.25 * np.sin(9.0 * t),
            np.where(np.arange(FIXTURE_SAMPLE_COUNT) % 2 == 0, 1.0, -0.5) + 0.05 * t,
        ],
        axis=0,
    )


def _assert_parity(histories: np.ndarray) -> None:
    """Assert tensErr gamma_method fields match pyerrors replica semantics."""
    expected = _pyerrors_reference(histories)
    actual = _normalize_result(_gamma_method(histories))
    np.testing.assert_allclose(actual["value"], expected["value"], rtol=PARITY_RTOL, atol=PARITY_ATOL)
    np.testing.assert_allclose(
        actual["stderr"], expected["stderr"], rtol=PARITY_RTOL, atol=PARITY_ATOL
    )
    np.testing.assert_array_equal(actual["window"], expected["window"])
    np.testing.assert_allclose(
        actual["tau_int"], expected["tau_int"], rtol=PARITY_RTOL, atol=PARITY_ATOL
    )


def test_gamma_method_matches_pyerrors_on_fixed_fixtures() -> None:
    """Check value, stderr, window, and tau_int parity on deterministic fixtures."""
    _assert_parity(_fixed_histories())


def test_gamma_method_matches_pyerrors_on_batched_fixtures() -> None:
    """Check pyerrors parity for inputs shaped ``(B, R, N)``."""
    histories = np.stack((_fixed_histories()[:3], _fixed_histories()[1:]), axis=0)

    _assert_parity(histories)


@pytest.mark.parametrize("rho", AR1_RHOS)
def test_gamma_method_matches_pyerrors_on_scalar_ar1_replicas(rho: float) -> None:
    """Check gamma_method parity on stationary scalar AR(1) replicas."""
    histories = scalar_ar1_replica_histories(
        AR1_REPLICA_COUNT,
        AR1_SAMPLE_COUNT,
        rho=rho,
        seed=20260614 + int(round(1000.0 * rho)),
    )
    _assert_parity(histories.numpy())
