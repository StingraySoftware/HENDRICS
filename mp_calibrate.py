from __future__ import division, print_function
import numpy as np
import os


def mp_default_nustar_rmf():
    print ("###############ATTENTION!!#####################")
    print ("")
    print ("Rmf not specified. Using default NuSTAR rmf.")
    print ("")
    print ("###############################################")
    return os.environ['CALDB'] + \
        "/data/nustar/fpm/cpf/rmf/nuAdet3_20100101v002.rmf"


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


def mp_calibrate(pis, rmf_file=None):
    '''Very rough calibration. Beware'''
    calp, calEmin, calEmax = mp_read_rmf(rmf_file)
    es = np.zeros(len(pis), dtype=np.float)
    for ic, c in enumerate(calp):
        good = pis == c
        if not np.any(good):
            continue
        es[good] = (calEmin[ic] + calEmax[ic]) / 2

    return es

if __name__ == '__main__':
    import argparse
    import cPickle as pickle
    description = 'Calibrate event light curves by associating the correct' + \
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
            outname = f.replace('.p', '_calib.p')

        # Read event file
        print ("Loading file %s..." % f)
        evdata = pickle.load(open(f))
        print ("Done.")
        pis = evdata['PI']

        es = mp_calibrate(pis, args.rmf)
        evdata['E'] = es
        print ('Saving calibrated data to %s' % outname)
        pickle.dump(evdata, open(outname, 'wb'))
