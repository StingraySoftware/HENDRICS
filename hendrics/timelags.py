from astropy import log

from .base import hen_root
from .io import load_pds, save_as_qdp


def main(args=None):
    import argparse

    from .base import _add_default_args

    description = "Read timelags from cross spectrum results and save them to a qdp file"
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

        lag = cross.time_lag()
        lag_err = None
        # ``time_lag`` returns either the lags alone, or a (lag, lag_err) pair,
        # depending on how much information the cross spectrum carries.
        if len(lag) == 2:
            lag, lag_err = lag
        out = hen_root(fname) + "_lags.qdp"
        save_as_qdp([cross.freq, lag], [None, lag_err], filename=out)
        filelist.append(out)

    return filelist
