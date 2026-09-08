# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""High ENergy Data Reduction Interface from the Command Shell."""

# The version file is written at build time by setuptools_scm; it is absent in
# a source checkout that has never been built or installed.
try:
    from ._version import version as __version__
except ImportError:
    __version__ = ""

# Workaround: import netCDF4 before everything else. This loads the HDF5
# library that netCDF4 uses and not something else.
try:
    import netCDF4 as nc

    HEN_FILE_EXTENSION = ".nc"
    HAS_NETCDF = True
except ImportError:
    HEN_FILE_EXTENSION = ".p"
    HAS_NETCDF = False

import warnings

import stingray

warnings.filterwarnings("ignore", message=".*Errorbars on cross.*")

from .compat import (
    HAS_NUMBA,
    array_take,
    float32,
    float64,
    int32,
    int64,
    njit,
    prange,
    vectorize,
)
