# -*- coding: utf-8 -*-
from __future__ import print_function, division
from .io import load_pds, HEN_FILE_EXTENSION
from .io import save_as_qdp
from astropy import log
from .base import hen_root


def main(args=None):
    import argparse
    description = ('Read timelags from cross spectrum results and save them'
                   ' to a qdp file')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')

    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING', type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    log.setLevel(args.loglevel)
    log.enable_warnings_logging()

    with log.log_to_file('HENlags.log'):
        filelist = []
        for fname in args.files:
            cross = load_pds(fname)

            lag, lag_err = cross.time_lag()
            out = hen_root(fname) + '_lags.qdp'
            save_as_qdp([cross.freq, lag], [None, lag_err], filename=out)
            filelist.append(out)

    return filelist
