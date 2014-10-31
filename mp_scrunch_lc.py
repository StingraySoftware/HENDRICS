from __future__ import division, print_function
from mp_lcurve import mp_scrunch_lightcurves


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-o", "--out", type=str, default="out_scrlc.p",
                        help='Output file')
    args = parser.parse_args()
    files = args.files

    mp_scrunch_lightcurves(files, args.out)
