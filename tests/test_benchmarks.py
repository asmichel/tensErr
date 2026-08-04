"""CPU benchmark tests comparing tensErr against pyerrors FFT."""

from __future__ import annotations

from collections.abc import Callable

import gc
import time

import numpy as np
import pytest

from tests.ar1 import scalar_ar1_replica_histories

# SCALAR_AR1_BENCHMARK_CASES specify replica count, sample count, and rho.
SCALAR_AR1_BENCHMARK_CASES = (
    (1, 262_144, 0.75),
    (32, 16_384, 0.75),
    (64, 65_536, 0.75),
)

# BENCHMARK_REPEATS keeps timing deterministic without making the suite excessive.
BENCHMARK_REPEATS = 3

# BENCHMARK_RATIO_LIMIT requires tensErr to be strictly faster than pyerrors FFT.
BENCHMARK_RATIO_LIMIT = 1.0

# BENCHMARK_ATOL allows benchmark warmup outputs to sanity-check agreement.
BENCHMARK_ATOL = 5.0e-10

# BENCHMARK_RTOL allows small float64 FFT roundoff differences in warmup outputs.
BENCHMARK_RTOL = 5.0e-8

# PYERRORS_ENSEMBLE_NAME is the pyerrors ensemble label for benchmark replicas.
PYERRORS_ENSEMBLE_NAME = "bench"

# PYERRORS_REPLICA_PREFIX namespaces each pyerrors replica inside the benchmark ensemble.
PYERRORS_REPLICA_PREFIX = "replica"


def _pyerrors_stderr(histories: np.ndarray) -> np.ndarray:
    """Return pyerrors FFT stderr for the benchmark history replica set."""
    pyerrors = pytest.importorskip("pyerrors")
    samples = [np.asarray(row, dtype=np.float64) for row in histories]
    names = [
        f"{PYERRORS_ENSEMBLE_NAME}|{PYERRORS_REPLICA_PREFIX}{i}"
        for i in range(histories.shape[0])
    ]
    obs = pyerrors.Obs(samples, names)
    obs.gamma_method(fft=True)
    return np.asarray(obs.dvalue, dtype=np.float64)


def _tenserr_stderr(histories: np.ndarray) -> np.ndarray:
    """Return tensErr stderr values from gamma_method."""
    torch = pytest.importorskip("torch")
    tenserr = pytest.importorskip("tensErr")
    result = tenserr.gamma_method(
        torch.as_tensor(histories, dtype=torch.float64, device="cpu")
    )
    if isinstance(result, dict):
        stderr = result.get("stderr", result.get("dvalue"))
        stderr = result.get("error", result.get("err")) if stderr is None else stderr
    elif isinstance(result, tuple) and not hasattr(result, "_fields"):
        stderr = result[1]
    else:
        stderr = getattr(result, "stderr", getattr(result, "dvalue", None))
        stderr = (
            getattr(result, "error", getattr(result, "err", None))
            if stderr is None
            else stderr
        )
    assert stderr is not None
    if hasattr(stderr, "detach"):
        stderr = stderr.detach().cpu().numpy()
    return np.asarray(stderr, dtype=np.float64)


def _minimum_seconds(
    callable_under_test: Callable[[np.ndarray], np.ndarray],
    histories: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return the minimum runtime and final value for repeated benchmark calls."""
    gc.collect()
    result = callable_under_test(histories)
    seconds = []
    for _ in range(BENCHMARK_REPEATS):
        gc.collect()
        start = time.perf_counter()
        result = callable_under_test(histories)
        seconds.append(time.perf_counter() - start)
    return min(seconds), result


@pytest.mark.parametrize(
    ("replica_count", "sample_count", "rho"),
    SCALAR_AR1_BENCHMARK_CASES,
)
def test_tenserr_cpu_outperforms_pyerrors_fft_for_scalar_ar1(
    replica_count: int,
    sample_count: int,
    rho: float,
    record_property: Callable[[str, object], None],
) -> None:
    """Require lower tensErr runtime for stationary scalar AR(1) histories."""
    seed = 20260614 + replica_count * 1_000_003 + sample_count
    histories = scalar_ar1_replica_histories(
        replica_count,
        sample_count,
        rho=rho,
        seed=seed,
    ).numpy()
    tenserr_seconds, tenserr_stderr = _minimum_seconds(_tenserr_stderr, histories)
    pyerrors_seconds, pyerrors_stderr = _minimum_seconds(_pyerrors_stderr, histories)
    np.testing.assert_allclose(
        tenserr_stderr, pyerrors_stderr, rtol=BENCHMARK_RTOL, atol=BENCHMARK_ATOL
    )
    ratio = tenserr_seconds / pyerrors_seconds
    record_property("tenserr_seconds", tenserr_seconds)
    record_property("pyerrors_fft_seconds", pyerrors_seconds)
    record_property("tenserr_over_pyerrors_ratio", ratio)
    record_property("rho", rho)
    print(
        f"shape={histories.shape} rho={rho:.3f} tensErr={tenserr_seconds:.6f}s "
        f"pyerrors_fft={pyerrors_seconds:.6f}s ratio={ratio:.3f}"
    )
    assert ratio < BENCHMARK_RATIO_LIMIT
