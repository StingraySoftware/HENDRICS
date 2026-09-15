"""Check the GPU benchmark code without a GPU.

The custom ``numba.cuda`` kernel in ``benchmarks/`` runs on the CPU through numba's
CUDA simulator. The simulator must be enabled before ``numba.cuda`` is first
imported, so the checks run in a separate Python process.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).parents[2] / "benchmarks" / "gpu_histogram_benchmark.py"

pytestmark = [
    pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmarks/ is only in the source tree"),
]

CHECKS = """
import importlib.util
import sys

import numpy as np
from stingray.fourier import avg_pds_from_timeseries

from hendrics.base import hist1d_numba_seq

spec = importlib.util.spec_from_file_location("benchmark", sys.argv[1])
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
assert benchmark.HAS_CUDA, "The CUDA simulator is not active"

# 1000 values: not a multiple of the thread block size, so some threads have
# tid >= size and must not read past the end of the array. The last values sit
# exactly on the bin edges, including the upper edge of the range.
rng = np.random.default_rng(0)
x = np.concatenate([rng.uniform(0, 1, 997), [0.0, 0.5, 1.0]])
result = benchmark.hist1d_numba_gpu(x, 17, [0.0, 1.0])
expected = hist1d_numba_seq(x, bins=17, ranges=[0.0, 1.0])
assert result.dtype == np.float64, result.dtype
assert np.array_equal(result, expected), (result, expected)

# The averaged power spectrum loops agree with stingray (NumPy standing in for CuPy)
segment_size, dt, length = 1.0, 0.01, 16.0
times = np.sort(rng.uniform(0, length, 5000))
gti = np.array([[0, length]])
pds = avg_pds_from_timeseries(times, gti, segment_size, dt, norm="none", silent=True)
expected = pds["unnorm_power"]
for func in (benchmark.avg_pds_unfused, benchmark.avg_pds_fused):
    assert np.allclose(func(times, gti, segment_size, dt, np), expected), func.__name__

# The bincount prototype of the --fast search gives exactly the sub-profiles of
# search_with_qffa_step (NumPy standing in for CuPy). Times are centered, as in
# search_with_qffa. Two events must be dropped: the last one, on the upper edge of
# the time range, and the one at -1e-17, whose phase rounds to exactly 1.0.
from hendrics.base import histogram2d
from hendrics.efsearch import _fast_phase, _fast_phase_fdot, _fast_phase_fddot

nbin, nprof = 8, 16
times = np.sort(np.concatenate([rng.uniform(-100, 100, 4997), [-1e-17, 0.0, 100.0]]))
d_times, d_slices = benchmark.qffa_upload(times, nprof, np)
for f, fdot, fddot in [(1.0, 0, 0), (1.0, 1e-3, 0), (1.0, 1e-3, 1e-6)]:
    if fddot != 0:
        phases = _fast_phase_fddot(times, f, fdot, fddot)
    elif fdot != 0:
        phases = _fast_phase_fdot(times, f, fdot)
    else:
        phases = _fast_phase(times, f)
    expected = histogram2d(
        phases, times, range=[[0, 1], [times[0], times[-1]]], bins=(nbin, nprof)
    ).T
    assert expected.sum() == times.size - 2, expected.sum()
    result = benchmark.qffa_profiles_bincount(d_times, d_slices, f, fdot, fddot, nbin, nprof, np)
    assert result.shape == (nprof, nbin), result.shape
    assert np.array_equal(result, expected), (f, fdot, fddot)

# The benchmark script itself runs
benchmark.main(
    ["--sizes", "2000", "--bins", "100", "--nprof", "32", "--segment-size", "1",
     "--dt", "0.01", "--n-segments", "4", "--qffa-nbin", "8", "--qffa-length", "200",
     "--repeat", "1"]
)
"""


def test_benchmark_code_with_cuda_simulator():
    pytest.importorskip("numba")
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")
    proc = subprocess.run(
        [sys.executable, "-c", CHECKS, str(BENCHMARK)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "agrees with CPU: False" not in proc.stdout, proc.stdout
