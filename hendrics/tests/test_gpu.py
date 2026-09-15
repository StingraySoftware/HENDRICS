import numpy as np
import pytest

from hendrics import gpu
from hendrics.base import histogram, histogram2d
from hendrics.gpu import (
    HAS_CUPY,
    histogram2d_gpu,
    histogram_gpu,
    resolve_backend,
)


@pytest.fixture
def fake_cupy(monkeypatch):
    """Replace the CuPy backend with a NumPy stand-in that counts host copies.

    There is no hardware simulator for CuPy, but the GPU code is a thin
    dispatch layer: a backend exposing the same functions with NumPy exercises
    argument handling, edge conventions, dtypes and host conversion.
    """
    calls = {"to_host": 0}

    def to_host(array):
        calls["to_host"] += 1
        return np.asarray(array)

    monkeypatch.setitem(
        gpu._BACKENDS,
        "cupy",
        {"available": lambda: True, "get_module": lambda: np, "to_host": to_host},
    )
    return calls


def test_resolve_backend_cpu():
    assert resolve_backend(use_gpu=False) == "cpu"


def test_resolve_backend_gpu_with_stand_in(fake_cupy):
    assert resolve_backend(use_gpu=True) == "cupy"


@pytest.mark.skipif(HAS_CUPY, reason="CuPy is installed")
def test_resolve_backend_raises_without_cupy():
    with pytest.raises(RuntimeError, match="CuPy"):
        resolve_backend(use_gpu=True)


def test_resolve_backend_raises_without_device(monkeypatch):
    monkeypatch.setitem(gpu._BACKENDS["cupy"], "available", lambda: False)
    with pytest.raises(RuntimeError, match="CuPy"):
        resolve_backend(use_gpu=True)


def test_histogram_gpu_raises_without_backend(monkeypatch):
    monkeypatch.setitem(gpu._BACKENDS["cupy"], "available", lambda: False)
    with pytest.raises(RuntimeError, match="CuPy"):
        histogram_gpu(np.zeros(10), bins=5, ranges=[0, 1])


@pytest.mark.parametrize("use_weights", [False, True])
def test_histogram_gpu_matches_cpu(fake_cupy, use_weights):
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 1000)
    weights = rng.uniform(0, 1, 1000) if use_weights else None
    expected = histogram(x, bins=17, ranges=[0.0, 1.0], weights=weights)
    result = histogram_gpu(x, bins=17, ranges=[0.0, 1.0], weights=weights)
    assert result.dtype == expected.dtype
    assert np.allclose(result, expected)
    assert fake_cupy["to_host"] == 1


def test_histogram_gpu_excludes_upper_edge(fake_cupy):
    # HENDRICS excludes values equal to the upper edge; numpy/cupy include them
    x = np.array([0.0, 0.25, 0.5, 1.0])
    expected = histogram(x, bins=4, ranges=[0.0, 1.0])
    result = histogram_gpu(x, bins=4, ranges=[0.0, 1.0])
    assert np.array_equal(result, expected)
    assert result.sum() == 3


@pytest.mark.parametrize("use_weights", [False, True])
def test_histogram2d_gpu_matches_cpu(fake_cupy, use_weights):
    rng = np.random.default_rng(2)
    times = np.sort(rng.uniform(0, 1000, 1000))
    phases = rng.uniform(0, 1, 1000)
    weights = rng.uniform(0, 1, 1000) if use_weights else None
    # Same call pattern as search_with_qffa_step: the last time is on the edge
    ranges = [[0, 1], [times[0], times[-1]]]
    expected = histogram2d(phases, times, bins=(16, 8), ranges=ranges, weights=weights)
    result = histogram2d_gpu(phases, times, bins=(16, 8), ranges=ranges, weights=weights)
    assert result.dtype == expected.dtype
    assert np.allclose(result, expected)
    assert fake_cupy["to_host"] == 1


def test_base_histogram_default_does_not_use_gpu(fake_cupy):
    x = np.random.default_rng(5).uniform(0, 1, 100)
    histogram(x, bins=5, range=[0.0, 1.0])
    histogram2d(x, x, bins=(5, 5), range=[[0.0, 1.0], [0.0, 1.0]])
    assert fake_cupy["to_host"] == 0


@pytest.mark.parametrize("use_weights", [False, True])
def test_base_histogram_use_gpu(fake_cupy, use_weights):
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 1, 1000)
    weights = rng.uniform(0, 1, 1000) if use_weights else None
    expected = histogram(x, bins=17, range=[0.0, 1.0], weights=weights)
    # Memory mapping only applies to host memory, and is ignored on the GPU
    result = histogram(
        x, bins=17, range=[0.0, 1.0], weights=weights, use_memmap=True, tmp=None, use_gpu=True
    )
    assert result.dtype == expected.dtype
    assert np.allclose(result, expected)
    assert fake_cupy["to_host"] == 1


@pytest.mark.parametrize("use_weights", [False, True])
def test_base_histogram2d_use_gpu(fake_cupy, use_weights):
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(0, 1000, 1000))
    phases = rng.uniform(0, 1, 1000)
    weights = rng.uniform(0, 1, 1000) if use_weights else None
    ranges = [[0, 1], [times[0], times[-1]]]
    expected = histogram2d(phases, times, bins=(16, 8), range=ranges, weights=weights)
    result = histogram2d(phases, times, bins=(16, 8), range=ranges, weights=weights, use_gpu=True)
    assert result.dtype == expected.dtype
    assert np.allclose(result, expected)
    assert fake_cupy["to_host"] == 1


def test_base_histogram_use_gpu_raises_without_backend(monkeypatch):
    monkeypatch.setitem(gpu._BACKENDS["cupy"], "available", lambda: False)
    with pytest.raises(RuntimeError, match="CuPy"):
        histogram(np.zeros(10), bins=5, range=[0, 1], use_gpu=True)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_histogram_gpu_real_cupy():
    if not gpu._BACKENDS["cupy"]["available"]():
        pytest.skip("No CUDA device visible")
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 100_000)
    expected = histogram(x, bins=1000, ranges=[0.0, 1.0])
    assert np.allclose(histogram_gpu(x, bins=1000, ranges=[0.0, 1.0]), expected)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_histogram2d_gpu_real_cupy():
    if not gpu._BACKENDS["cupy"]["available"]():
        pytest.skip("No CUDA device visible")
    rng = np.random.default_rng(4)
    times = np.sort(rng.uniform(0, 1000, 100_000))
    phases = rng.uniform(0, 1, 100_000)
    ranges = [[0, 1], [times[0], times[-1]]]
    expected = histogram2d(phases, times, bins=(32, 64), ranges=ranges)
    result = histogram2d_gpu(phases, times, bins=(32, 64), ranges=ranges)
    assert np.allclose(result, expected)
