from astropy import log

from .base import hen_root
from .io import load_pds, save_as_qdp


def main(args=None):
    import argparse

    from .base import _add_default_args

    description = "Read timelags from cross spectrum results and save them" " to a qdp file"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs="+")

    _add_default_args(parser, ["loglevel", "debug"])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    filelist = []
    for fname in args.files:
        cross = load_pds(fname)

        lag, lag_err = cross.time_lag()
        out = hen_root(fname) + "_lags.qdp"
        save_as_qdp([cross.freq, lag], [None, lag_err], filename=out)
        filelist.append(out)

    return filelist
