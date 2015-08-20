# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calibrate event lists by looking in rmf files."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import load_events, save_events
import numpy as np
import os
import logging


def default_nustar_rmf():
    """Look for the default rmf file in the CALDB.

    The CALDB environment variable has to point to the correct location of
    the NuSTAR CALDB

    .. note:: The calibration might change in the future. The hardcoded file
              name will be eventually replaced with a smarter choice based
              on observing time
    """
    logging.warning("Rmf not specified. Using default NuSTAR rmf.")
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
        The MaLTPyNT file containing the events
    outname : str
        The output file

    Other Parameters
    ----------------
    rmf_file : str
        The rmf file used to read the calibration. If None or not specified,
        the one given by default_nustar_rmf() is used.
    """
    # Read event file
    logging.info("Loading file %s..." % fname)
    evdata = load_events(fname)
    logging.info("Done.")
    pis = evdata['PI']

    es = read_calibration(pis, rmf_file)
    evdata['E'] = es
    logging.info('Saving calibrated data to %s' % outname)
    save_events(evdata, outname)


if __name__ == '__main__':  # pragma: no cover
    import sys
    import subprocess as sp

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MPcalibrate'] + args)
