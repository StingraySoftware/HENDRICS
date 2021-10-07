# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This is proposed as an Astropy affiliated package."""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *

# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    # Workaround: import netCDF4 before everything else. This loads the HDF5
    # library that netCDF4 uses and not something else.

    try:
        import netCDF4 as nc

        HEN_FILE_EXTENSION = ".nc"
        HAS_NETCDF = True
    except ImportError:
        HEN_FILE_EXTENSION = ".p"
        HAS_NETCDF = False
        pass

    import stingray
    import warnings

    warnings.filterwarnings("ignore", message=".*Errorbars on cross.*")

    from .compat import (
        prange,
        array_take,
        HAS_NUMBA,
        njit,
        vectorize,
        float32,
        float64,
        int32,
        int64,
        _MonkeyPatchedEventList,
        power_confidence_limits,
        power_upper_limit,
        pf_from_ssig,
        pf_from_a,
        pf_upper_limit,
        a_from_pf,
        a_from_ssig,
        ssig_from_a,
        ssig_from_pf,
    )

    stingray.events.EventList = _MonkeyPatchedEventList
    try:
        from stingray.stats import pf_upper_limit, power_confidence_limits

        power_confidence_limits(50, alpha=0.16)
    except (ImportError, TypeError):
        stingray.stats.power_confidence_limits = power_confidence_limits
        stingray.stats.power_upper_limit = power_upper_limit
        stingray.stats.pf_from_ssig = pf_from_ssig
        stingray.stats.pf_from_a = pf_from_a
        stingray.stats.pf_upper_limit = pf_upper_limit
        stingray.stats.a_from_pf = a_from_pf
        stingray.stats.a_from_ssig = a_from_ssig
        stingray.stats.ssig_from_a = ssig_from_a
        stingray.stats.ssig_from_pf = ssig_from_pf
