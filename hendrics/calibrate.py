# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calibrate event lists by looking in rmf files."""

import os
import warnings
import numpy as np
from astropy import log

from .base import get_file_extension
from .io import load_events, save_events
from .io import HEN_FILE_EXTENSION


def default_nustar_rmf():
    """Look for the default rmf file in the CALDB.

    The CALDB environment variable has to point to the correct location of
    the NuSTAR CALDB

    .. note:: The calibration might change in the future. The hardcoded file
              name will be eventually replaced with a smarter choice based
              on observing time
    """
    warnings.warn("Rmf not specified. Using default NuSTAR rmf.")
    rmf = "data/nustar/fpm/cpf/rmf/nuAdet3_20100101v002.rmf"
    path = rmf.split("/")
    newpath = os.path.join(os.environ["CALDB"], *path)
    return newpath


def read_rmf(rmf_file=None):
    """Load RMF info.

    .. note:: Preliminary: only EBOUNDS are read.

    Parameters
    ----------
    rmf_file : str
        The rmf file used to read the calibration. If None or not specified,
        the one given by default_nustar_rmf() is used.

    Returns
    -------
    pis : array-like
        the PI channels
    e_mins : array-like
        the lower energy bound of each PI channel
    e_maxs : array-like
        the upper energy bound of each PI channel
    """
    from astropy.io import fits as pf

    if rmf_file is None or rmf_file == "":
        rmf_file = default_nustar_rmf()

    lchdulist = pf.open(rmf_file, checksum=True)
    lchdulist.verify("warn")
    lctable = lchdulist["EBOUNDS"].data
    pis = np.array(lctable.field("CHANNEL"))
    e_mins = np.array(lctable.field("E_MIN"))
    e_maxs = np.array(lctable.field("E_MAX"))
    lchdulist.close()
    return pis, e_mins, e_maxs


def read_calibration(pis, rmf_file=None):
    """Read the energy channels corresponding to the given PI channels.

    Parameters
    ----------
    pis : array-like
        The channels to lookup in the rmf

    Other Parameters
    ----------------
    rmf_file : str
        The rmf file used to read the calibration. If None or not specified,
        the one given by default_nustar_rmf() is used.
    """
    calp, calEmin, calEmax = read_rmf(rmf_file)
    es = np.zeros(len(pis), dtype=float)
    for ic, c in enumerate(calp):
        good = pis == c
        if not np.any(good):
            continue
        es[good] = (calEmin[ic] + calEmax[ic]) / 2

    return es


def rough_calibration(pis, mission):
    """

    Parameters
    ----------
    pis: float or array of floats
        PI channels in data
    mission: str
        Mission name

    Returns
    -------
    energies : float or array of floats
        Energy values

    Examples
    --------
    >>> rough_calibration(0, 'nustar')
    1.6
    >>> rough_calibration(0.0, 'ixpe')
    0.0
    >>> # It's case-insensitive
    >>> rough_calibration(1200, 'XMm')
    1.2
    >>> rough_calibration(10, 'asDf')
    Traceback (most recent call last):
        ...
    ValueError: Mission asdf not recognized
    >>> rough_calibration(100, 'nicer')
    1.0
    """
    if mission.lower() == "nustar":
        return pis * 0.04 + 1.6
    elif mission.lower() == "xmm":
        return pis * 0.001
    elif mission.lower() == "nicer":
        return pis * 0.01
    elif mission.lower() == "ixpe":
        return pis / 375 * 15
    raise ValueError(f"Mission {mission.lower()} not recognized")


def calibrate(fname, outname, rmf_file=None, rough=False):
    """Do calibration of an event list.

    Parameters
    ----------
    fname : str
        The HENDRICS file containing the events
    outname : str
        The output file

    Other Parameters
    ----------------
    rmf_file : str
        The rmf file used to read the calibration. If None or not specified,
        the one given by default_nustar_rmf() is used.
    """
    # Read event file
    log.info("Loading file %s..." % fname)
    evdata = load_events(fname)
    log.info("Done.")
    pis = evdata.pi

    if rough:
        cal_pis = evdata.pi
        if hasattr(evdata, "cal_pi") and evdata.cal_pi is not None:
            cal_pis = evdata.cal_pi
        es = rough_calibration(cal_pis, evdata.mission)
    else:
        if evdata.mission.lower() == "xmm":
            raise RuntimeError(
                "Calibration for XMM should work out-of-the box in "
                "HENreadevents. Running HENcalibrate with the --rmf option is"
                " known to produce wrong results in XMM"
            )
        es = read_calibration(pis, rmf_file)

    evdata.energy = es
    log.info("Saving calibrated data to %s" % outname)
    save_events(evdata, outname)


def _calib_wrap(args):
    f, outname, rmf, rough = args
    return calibrate(f, outname, rmf, rough)


def main(args=None):
    """Main function called by the `HENcalibrate` command line script."""
    import argparse
    from multiprocessing import Pool
    from .base import _add_default_args

    description = (
        "Calibrate clean event files by associating the correct "
        "energy to each PI channel. Uses either a specified rmf "
        "file or (for NuSTAR only) an rmf file from the CALDB"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-r",
        "--rmf",
        help="rmf file used for calibration. Not working with XMM data",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--rough",
        help="Rough calibration, without rmf file "
        "(only for NuSTAR, XMM, and NICER). Only for compatibility purposes. "
        "This is done automatically by HENreadevents",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite; default: no",
        default=False,
        action="store_true",
    )
    _add_default_args(parser, ["nproc", "loglevel", "debug"])

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    with log.log_to_file("HENcalibrate.log"):
        funcargs = []
        for i_f, f in enumerate(files):
            outname = f
            if args.overwrite is False:
                label = "_calib"
                if args.rough:
                    label = "_rough_calib"
                outname = f.replace(get_file_extension(f), label + HEN_FILE_EXTENSION)
            funcargs.append([f, outname, args.rmf, args.rough])

        if os.name == "nt" or args.nproc == 1:
            [_calib_wrap(fa) for fa in funcargs]
        else:
            pool = Pool(processes=args.nproc)
            for i in pool.imap_unordered(_calib_wrap, funcargs):
                pass
            pool.close()
