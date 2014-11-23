from __future__ import division, print_function
from mp_lcurve import mp_scrunch_lightcurves
from mp_io import MP_FILE_EXTENSION


if __name__ == '__main__':
    import argparse
    description = \
        'Sums lightcurves from different instruments or energy ranges'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-o", "--out", type=str,
                        default="out_scrlc" + MP_FILE_EXTENSION,
                        help='Output file')
    args = parser.parse_args()
    files = args.files

    mp_scrunch_lightcurves(files, args.out)
