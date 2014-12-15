from __future__ import division, print_function
from mp_io import mp_load_events, mp_save_events
from mp_io import mp_get_file_extension, MP_FILE_EXTENSION
import numpy as np
import os


def mp_default_nustar_rmf():
    print ("###############ATTENTION!!#####################")
    print ("")
    print ("Rmf not specified. Using default NuSTAR rmf.")
    print ("")
    print ("###############################################")
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
    print ("Loading file %s..." % fname)
    evdata = mp_load_events(fname)
    print ("Done.")
    pis = evdata['PI']

    es = mp_read_calibration(pis, rmf)
    evdata['E'] = es
    print ('Saving calibrated data to %s' % outname)
    mp_save_events(evdata, outname)


if __name__ == '__main__':
    import argparse
    description = 'Calibrates clean event files by associating the correct' + \
        ' energy to each PI channel. Uses either a specified rmf file or' + \
        ' (for NuSTAR only) an rmf file from the CALDB'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-r", "--rmf", help="rmf file used for calibration",
                        default=None, type=str)
    parser.add_argument("-o", "--overwrite",
                        help="Overwrite; default: no",
                        default=False,
                        action="store_true")
    args = parser.parse_args()

    files = args.files

    for i_f, f in enumerate(files):
        outname = f
        if args.overwrite is False:
            outname = f.replace(mp_get_file_extension(f), '_calib' +
                                MP_FILE_EXTENSION)
        mp_calibrate(f, outname, args.rmf)
