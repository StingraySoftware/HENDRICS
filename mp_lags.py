from __future__ import division, print_function
#from mp_fspec import mp_calc_lags


def mp_lags_from_spectra(cpdsfile, pds1file, pds2file):
    print ('mp_lags_from_spectra not yet implemented')
    return


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
