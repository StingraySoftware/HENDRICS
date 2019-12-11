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

    import warnings
    try:
        import netCDF4 as nc

        HEN_FILE_EXTENSION = '.nc'
        HAS_NETCDF = True
    except ImportError:
        msg = "Warning! NetCDF is not available. Using pickle format."
        warnings.warn(msg)
        HEN_FILE_EXTENSION = '.p'
        HAS_NETCDF = False
        pass

    from . import base
    from . import binary
    from . import calibrate
    from . import colors
    from . import create_gti
    from . import efsearch
    from . import exposure
    from . import exvar
    from . import fake
    from . import fspec
    from . import io
    from . import lcurve
    from . import modeling
    from . import plot
    from . import read_events
    from . import rebin
    from . import save_as_xspec
    from . import sum_fspec
    from . import timelags
    from . import varenergy
