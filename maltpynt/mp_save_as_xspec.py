from __future__ import division, print_function
from mp_io import mp_get_file_type
import numpy as np
from mp_io import mp_get_file_extension


def mp_save_as_xspec(fname):
    ftype, contents = mp_get_file_type(fname)

    outname = fname.replace(mp_get_file_extension(fname), '_xsp.dat')

    if 'freq' in contents.keys():
        freq = contents['freq']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = freq[1] - freq[0]

        np.savetxt(outname, np.transpose([freq - df / 2,
                                          freq + df / 2,
                                          pds.real * df,
                                          epds * df]))
    elif 'flo' in contents.keys():
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
    args = parser.parse_args()
    fnames = args.files
    for f in fnames:
        mp_save_as_xspec(f)
