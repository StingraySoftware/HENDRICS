"""The Numba functions of HENDRICS are cached on disk, and not compiled again at each run."""

import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from hendrics.base import HAS_NUMBA

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="Caching is a Numba feature")

MODULES = ["hendrics.base", "hendrics.efsearch", "hendrics.ffa", "hendrics.ml_timing"]


def _njit_functions(module_name):
    from numba.core.dispatcher import Dispatcher

    module = importlib.import_module(module_name)
    return {
        name: obj
        for name, obj in vars(module).items()
        if isinstance(obj, Dispatcher) and obj.py_func.__module__ == module_name
    }


@pytest.mark.parametrize("module_name", MODULES)
def test_all_njit_functions_are_cached(module_name):
    functions = _njit_functions(module_name)
    assert functions, f"No njit functions found in {module_name}"
    # Without caching, Numba reports no cache path
    not_cached = sorted(name for name, func in functions.items() if func.stats.cache_path is None)
    assert not_cached == []


# Runs a small fast search on the CPU, going through all the phase formulas, and reports which
# functions were loaded from the cache
SEARCH = """
import json
import sys

import numpy as np

from hendrics import base, efsearch

rng = np.random.default_rng(1)
times = np.sort(rng.uniform(0, 1000, 20000))
stats = []
for fdot, fddot in [(0, 0), (1e-7, 0), (1e-7, 1e-10)]:
    _, _, stat, _, _, _ = efsearch.search_with_qffa(
        times, 1, 1.01, fdot=fdot, fddot=fddot, nbin=16, n=2, silent=True
    )
    stats.append(stat)
efsearch._fast_step_constants(32, 16, 2)
np.save(sys.argv[1], np.array(stats))

functions = {
    "_fast_step": efsearch._fast_step,
    "_fast_step_constants": efsearch._fast_step_constants,
    "_fast_phase": efsearch._fast_phase,
    "_fast_phase_fdot": efsearch._fast_phase_fdot,
    "_fast_phase_fddot": efsearch._fast_phase_fddot,
    "_hist2d_numba_seq": base._hist2d_numba_seq,
}
print(json.dumps({name: sum(func.stats.cache_hits.values()) for name, func in functions.items()}))
"""


def test_fast_search_is_loaded_from_cache_in_a_new_process(tmp_path):
    # A cache folder of our own: the first process starts with an empty cache
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"))
    hits, stats = [], []
    for run in range(2):
        out = tmp_path / f"stats_{run}.npy"
        proc = subprocess.run(
            [sys.executable, "-c", SEARCH, str(out)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        hits.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        stats.append(np.load(out))

    assert all(count == 0 for count in hits[0].values()), hits[0]
    not_from_cache = sorted(name for name, count in hits[1].items() if count == 0)
    assert not_from_cache == []
    # The code loaded from the cache gives exactly the same results
    assert np.array_equal(stats[0], stats[1])


# Imports what HENzsearch imports, and lists the Numba code already compiled
IMPORT_ONLY = """
import sys

import numpy as np
from numba.core.dispatcher import Dispatcher
from numba.np.ufunc.dufunc import DUFunc

import hendrics.efsearch

compiled = []
for module_name, module in list(sys.modules.items()):
    if not module_name.startswith("hendrics") or module is None:
        continue
    for name, obj in vars(module).items():
        if isinstance(obj, Dispatcher) and obj.signatures:
            compiled.append(f"{module_name}.{name}")
        # @vectorize with signatures compiles on the spot, making a DUFunc (or a NumPy
        # ufunc, with older Numba versions) with compiled loops
        elif (
            isinstance(obj, (DUFunc, np.ufunc))
            and obj.__name__ == name
            and getattr(np, name, None) is not obj
            and obj.types
        ):
            compiled.append(f"{module_name}.{name}")
print(sorted(set(compiled)))
"""


def test_importing_efsearch_compiles_no_numba_code():
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_ONLY], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip().splitlines()[-1] == "[]"
