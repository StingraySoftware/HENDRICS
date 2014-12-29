from __future__ import division, print_function
from .mp_io import mp_get_file_type
import numpy as np
from .mp_io import mp_get_file_extension
import logging


def mp_save_as_xspec(fname):
    ftype, contents = mp_get_file_type(fname)

    outname = fname.replace(mp_get_file_extension(fname), '_xsp.dat')

    if 'freq' in list(contents.keys()):
        freq = contents['freq']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = freq[1] - freq[0]

        np.savetxt(outname, np.transpose([freq - df / 2,
                                          freq + df / 2,
                                          pds.real * df,
                                          epds * df]))
    elif 'flo' in list(contents.keys()):
        ftype = ftype.replace('reb', '')
        flo = contents['flo']
        fhi = contents['fhi']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = fhi - flo
        np.savetxt(outname, np.transpose([flo, fhi,
                                          pds.real * df,
                                          epds * df]))
    else:
        raise Exception('File type not recognized')


if __name__ == '__main__':
    import argparse
    description = ('Save a frequency spectrum in a qdp file that can be read '
                   'by flx2xsp and produce a XSpec-compatible spectrum file')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    args = parser.parse_args()
    files = args.files
    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MP2xpec.log', level=numeric_level,
                        filemode='w')

    for f in files:
        mp_save_as_xspec(f)
