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

    import stingray.utils
    from stingray.events import EventList

    # Patch stingray.utils
    def _root_squared_mean(array):
        import numpy as np

        array = np.asarray(array)
        return np.sqrt(np.sum(array ** 2)) / len(array)

    stingray.utils._root_squared_mean = _root_squared_mean

    class MonkeyPatchedEventList(EventList):
        def apply_mask(self, mask, inplace=False):  # pragma: no cover
            """For compatibility with old stingray version.

            Examples
            --------
            >>> evt = MonkeyPatchedEventList(time=[0, 1, 2])
            >>> newev0 = evt.apply_mask([True, True, False], inplace=False);
            >>> newev1 = evt.apply_mask([True, True, False], inplace=True);
            >>> np.allclose(newev0.time, [0, 1])
            True
            >>> np.allclose(newev1.time, [0, 1])
            True
            >>> evt is newev1
            True
            """
            import copy
            if inplace:
                new_ev = self
            else:
                new_ev = copy.deepcopy(self)
            for attr in "time", "energy", "pi", "cal_pi":
                if hasattr(new_ev, attr) and getattr(new_ev, attr) is not None:
                    setattr(new_ev, attr, getattr(new_ev, attr)[mask])
            return new_ev

    try:
        e = EventList(time=[1, 2, 3])
        e.energy = None
        e.apply_mask([True, True, False])
        print(e)
    except (TypeError, AttributeError):
        print("Monkey patching Eventlist")
        stingray.events.EventList = MonkeyPatchedEventList
