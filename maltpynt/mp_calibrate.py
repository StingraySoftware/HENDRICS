# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .mp_io import mp_load_events, mp_save_events
import numpy as np
import os
import logging


def mp_default_nustar_rmf():
    logging.warning("Rmf not specified. Using default NuSTAR rmf.")
    rmf = "data/nustar/fpm/cpf/rmf/nuAdet3_20100101v002.rmf"
    path = rmf.split('/')
    newpath = os.path.join(os.environ['CALDB'], *path)
    return newpath


def mp_read_rmf(rmf_file=None):
    '''Loads RMF info
    preliminary: only EBOUNDS
    '''
    from astropy.io import fits as pf

    if rmf_file is None or rmf_file == '':
        rmf_file = mp_default_nustar_rmf()

    lchdulist = pf.open(rmf_file, checksum=True)
    lchdulist.verify('warn')
    lctable = lchdulist['EBOUNDS'].data
    pis = np.array(lctable.field('CHANNEL'))
    e_mins = np.array(lctable.field('E_MIN'))
    e_maxs = np.array(lctable.field('E_MAX'))
    lchdulist.close()
    return pis, e_mins, e_maxs


def mp_read_calibration(pis, rmf_file=None):
    '''Very rough calibration. Beware'''
    calp, calEmin, calEmax = mp_read_rmf(rmf_file)
    es = np.zeros(len(pis), dtype=np.float)
    for ic, c in enumerate(calp):
        good = pis == c
        if not np.any(good):
            continue
        es[good] = (calEmin[ic] + calEmax[ic]) / 2

    return es


def mp_calibrate(fname, outname, rmf=None):
    '''Do calibration'''
    # Read event file
    logging.info("Loading file %s..." % fname)
    evdata = mp_load_events(fname)
    logging.info("Done.")
    pis = evdata['PI']

    es = mp_read_calibration(pis, rmf)
    evdata['E'] = es
    logging.info('Saving calibrated data to %s' % outname)
    mp_save_events(evdata, outname)


if __name__ == '__main__':
    import sys
    import subprocess as sp

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MPcalibrate'] + args)
