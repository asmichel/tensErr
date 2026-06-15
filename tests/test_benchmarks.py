"""CPU benchmark tests comparing torch_uwerr against pyerrors FFT."""

from __future__ import annotations

from collections.abc import Callable

import gc
import time

import numpy as np
import pytest

# BENCHMARK_SHAPES are the requested float64 history layouts.
BENCHMARK_SHAPES = ((32, 16_384), (64, 65_536))

# BENCHMARK_RHO gives the benchmark histories realistic positive autocorrelation.
BENCHMARK_RHO = 0.75

# BENCHMARK_REPEATS keeps timing deterministic without making the suite excessive.
BENCHMARK_REPEATS = 3

# BENCHMARK_RATIO_LIMIT fails when pyerrors FFT is more than about 2x faster.
BENCHMARK_RATIO_LIMIT = 2.0

# BENCHMARK_ATOL allows benchmark warmup outputs to sanity-check agreement.
BENCHMARK_ATOL = 5.0e-10

# BENCHMARK_RTOL allows small float64 FFT roundoff differences in warmup outputs.
BENCHMARK_RTOL = 5.0e-8

# PYERRORS_ENSEMBLE_NAME is the pyerrors ensemble label for benchmark replicas.
PYERRORS_ENSEMBLE_NAME = "bench"

# PYERRORS_REPLICA_PREFIX namespaces each pyerrors replica inside the benchmark ensemble.
PYERRORS_REPLICA_PREFIX = "replica"


def _benchmark_histories(shape: tuple[int, int]) -> np.ndarray:
    """Return deterministic float64 AR(1) histories for a benchmark shape."""
    chain_count, sample_count = shape
    seed = 20260614 + chain_count * 1_000_003 + sample_count
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(shape)
    histories = np.empty_like(innovations)
    histories[:, 0] = innovations[:, 0]
    innovation_scale = np.sqrt(1.0 - BENCHMARK_RHO * BENCHMARK_RHO)
    for t in range(1, sample_count):
        histories[:, t] = (
            BENCHMARK_RHO * histories[:, t - 1] + innovation_scale * innovations[:, t]
        )
    return histories


def _pyerrors_stderr(histories: np.ndarray) -> np.ndarray:
    """Return pyerrors FFT stderr for the benchmark history chain set."""
    pyerrors = pytest.importorskip("pyerrors")
    samples = [np.asarray(row, dtype=np.float64) for row in histories]
    names = [
        f"{PYERRORS_ENSEMBLE_NAME}|{PYERRORS_REPLICA_PREFIX}{i}"
        for i in range(histories.shape[0])
    ]
    obs = pyerrors.Obs(samples, names)
    obs.gamma_method(fft=True)
    return np.asarray(obs.dvalue, dtype=np.float64)


def _torch_uwerr_stderr(histories: np.ndarray) -> np.ndarray:
    """Return torch_uwerr stderr values from gamma_method_mean."""
    torch = pytest.importorskip("torch")
    torch_uwerr = pytest.importorskip("torch_uwerr")
    result = torch_uwerr.gamma_method_mean(
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


@pytest.mark.parametrize("shape", BENCHMARK_SHAPES)
def test_torch_uwerr_cpu_is_not_slower_than_pyerrors_fft_by_more_than_2x(
    shape: tuple[int, int],
    record_property: Callable[[str, object], None],
) -> None:
    """Report CPU timings and fail if pyerrors FFT is more than about 2x faster."""
    histories = _benchmark_histories(shape)
    torch_seconds, torch_stderr = _minimum_seconds(_torch_uwerr_stderr, histories)
    pyerrors_seconds, pyerrors_stderr = _minimum_seconds(_pyerrors_stderr, histories)
    np.testing.assert_allclose(
        torch_stderr, pyerrors_stderr, rtol=BENCHMARK_RTOL, atol=BENCHMARK_ATOL
    )
    ratio = torch_seconds / pyerrors_seconds
    record_property("torch_uwerr_seconds", torch_seconds)
    record_property("pyerrors_fft_seconds", pyerrors_seconds)
    record_property("torch_over_pyerrors_ratio", ratio)
    print(
        f"shape={shape} torch_uwerr={torch_seconds:.6f}s "
        f"pyerrors_fft={pyerrors_seconds:.6f}s ratio={ratio:.3f}"
    )
    assert ratio <= BENCHMARK_RATIO_LIMIT
