# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This is proposed as an Astropy affiliated package."""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import base
    from . import calibrate
    from . import create_gti
    from . import exposure
    from . import fake
    from . import fspec
    from . import io
    from . import lcurve
    from . import plot
    from . import read_events
    from . import rebin
    from . import save_as_xspec
    from . import sum_fspec
