# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calibrate event lists by looking in rmf files."""

import warnings
from .io import load_events, save_events, get_file_extension
from .io import HEN_FILE_EXTENSION
import numpy as np
import os
from astropy import log
from astropy.logger import AstropyUserWarning


def default_nustar_rmf():
    """Look for the default rmf file in the CALDB.

    The CALDB environment variable has to point to the correct location of
    the NuSTAR CALDB

    .. note:: The calibration might change in the future. The hardcoded file
              name will be eventually replaced with a smarter choice based
              on observing time
    """
    warnings.warn("Rmf not specified. Using default NuSTAR rmf.", AstropyUserWarning)
    rmf = "data/nustar/fpm/cpf/rmf/nuAdet3_20100101v002.rmf"
    path = rmf.split('/')
    newpath = os.path.join(os.environ['CALDB'], *path)
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

    if rmf_file is None or rmf_file == '':
        rmf_file = default_nustar_rmf()

    lchdulist = pf.open(rmf_file, checksum=True)
    lchdulist.verify('warn')
    lctable = lchdulist['EBOUNDS'].data
    pis = np.array(lctable.field('CHANNEL'))
    e_mins = np.array(lctable.field('E_MIN'))
    e_maxs = np.array(lctable.field('E_MAX'))
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
    es = np.zeros(len(pis), dtype=np.float)
    for ic, c in enumerate(calp):
        good = pis == c
        if not np.any(good):
            continue
        es[good] = (calEmin[ic] + calEmax[ic]) / 2

    return es


def calibrate(fname, outname, rmf_file=None):
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

    es = read_calibration(pis, rmf_file)
    evdata.energy = es
    log.info('Saving calibrated data to %s' % outname)
    save_events(evdata, outname)


def _calib_wrap(args):
    f, outname, rmf = args
    return calibrate(f, outname, rmf)


def main(args=None):
    """Main function called by the `HENcalibrate` command line script."""
    import argparse
    from multiprocessing import Pool

    description = ('Calibrate clean event files by associating the correct '
                   'energy to each PI channel. Uses either a specified rmf '
                   'file or (for NuSTAR only) an rmf file from the CALDB')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-r", "--rmf", help="rmf file used for calibration",
                        default=None, type=str)
    parser.add_argument("-o", "--overwrite",
                        help="Overwrite; default: no",
                        default=False,
                        action="store_true")
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--nproc",
                        help=("Number of processors to use"),
                        default=1,
                        type=int)

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = 'DEBUG'

    log.setLevel(args.loglevel)


    with log.log_to_file('HENcalibrate.log'):
        funcargs = []
        for i_f, f in enumerate(files):
            outname = f
            if args.overwrite is False:
                outname = f.replace(get_file_extension(f), '_calib' +
                                    HEN_FILE_EXTENSION)
            funcargs.append([f, outname, args.rmf])

        if os.name == 'nt' or args.nproc == 1:
            [_calib_wrap(fa) for fa in funcargs]
        else:
            pool = Pool(processes=args.nproc)
            for i in pool.imap_unordered(_calib_wrap, funcargs):
                pass
            pool.close()
