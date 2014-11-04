from __future__ import division, print_function
from mp_fspec import mp_calc_lags, mp_read_fspec


def mp_lags_from_spectra(cpdsfile, pds1file, pds2file):
    print ('mp_lags_from_spectra Needs testing')

    ftype, cfreq, cpds, ecpds, nchunks, rebin = mp_read_fspec(cpdsfile)
    ftype, p1freq, pds1, epds1, nchunks, rebin = mp_read_fspec(pds1file)
    ftype, p2freq, pds2, epds2, nchunks, rebin = mp_read_fspec(pds2file)

    assert len(cpds) == len(pds1), 'Files are not compatible'
    assert len(cpds) == len(pds2), 'Files are not compatible'

    lags, elags = mp_calc_lags(cfreq, cpds, pds1, pds2, nchunks, rebin)
    return lags, elags


if __name__ == '__main__':
    import argparse
    description = 'Calculates time lags from the cross power spectrum and' + \
        ' the power spectra of the two channels'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="Three files: the cross spectrum" +
                        " and the two power spectra", nargs='+')
    args = parser.parse_args()

    if len(args.files) != 3:
        raise Exception('Invalid number of arguments')
    cfile, p1file, p2file = args.files
    mp_lags_from_spectra(cfile, p1file, p2file)
