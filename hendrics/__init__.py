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
    from .compat import (
        _MonkeyPatchedEventList,
        filter_for_deadtime,
        get_deadtime_mask,
        read_mission_info,
        _case_insensitive_search_in_list,
        get_key_from_mission_info,
    )

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
    )

    try:
        e = stingray.events.EventList(time=[1, 2, 3])
        e.energy = None
        e.apply_mask([True, True, False])
    except (TypeError, AttributeError):
        stingray.events.EventList = _MonkeyPatchedEventList
        stingray.filters.filter_for_deadtime = filter_for_deadtime
        stingray.filters.get_deadtime_mask = get_deadtime_mask
        stingray.io.read_mission_info = read_mission_info
