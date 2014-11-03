from __future__ import division, print_function
from mp_base import mp_get_file_type
import numpy as np


def mp_save_as_xspec(fname):
    ftype, contents = mp_get_file_type(fname)

    outname = fname.replace('.p', '_xsp.qdp')

    if 'freq' in contents.keys():
        freq = contents['freq']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = freq[1] - freq[0]

        np.savetxt(outname, np.transpose([freq.real - df / 2,
                                          freq.real + df / 2,
                                          pds.real * df,
                                          epds.real * df]))
    elif 'flo' in contents.keys():
        ftype = ftype.replace('reb', '')
        flo = contents['flo']
        fhi = contents['fhi']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = fhi - flo
        np.savetxt(outname, np.transpose([flo.real, fhi.real,
                                          pds.real * df,
                                          epds.real * df]))
    else:
        raise Exception('File type not recognized')


if __name__ == '__main__':
    import sys
    fnames = sys.argv[1:]
    for f in fnames:
        mp_save_as_xspec(f)
