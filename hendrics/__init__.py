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

        HEN_FILE_EXTENSION = '.nc'
        HAS_NETCDF = True
    except ImportError:
        HEN_FILE_EXTENSION = '.p'
        HAS_NETCDF = False
        pass

    from . import base
    from . import binary
    from . import calibrate
    from . import colors
    from . import conftest
    from . import create_gti
    from . import cython_version
    from . import efsearch
    from . import exposure
    from . import exvar
    from . import fake
    from . import fold
    from . import fspec
    from . import io
    from . import lcurve
    from . import modeling
    from . import phaseogram
    from . import phasetag
    from . import plot
    from . import read_events
    from . import rebin
    from . import save_as_xspec
    from . import setup_package
    from . import sum_fspec
    from . import timelags
    from . import varenergy
    from . import version
