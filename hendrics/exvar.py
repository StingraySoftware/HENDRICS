#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:55:47 2017

@author: marta
"""

from astropy import log
from stingray.utils import excess_variance
from .io import load_lcurve
from .io import save_as_qdp
from .base import hen_root


def fvar(lc):
    return excess_variance(lc, normalization="fvar")


def excvar_none(lc):
    return excess_variance(lc, normalization="none")


def excvar_norm(lc):
    return excess_variance(lc, normalization="norm_xs")


def main(args=None):
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Calculate excess variance in light curve chunks"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-c",
        "--chunk-length",
        type=float,
        default=20,
        help="Length in seconds of the light curve chunks",
    )
    parser.add_argument(
        "--fraction-step",
        type=float,
        default=0.5,
        help="If the step is not a full chunk_length but less,"
        "this indicates the ratio between step step and"
        " `chunk_length`",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="excvar",
        help="Choose between fvar, excvar and norm_excvar "
        "normalization, referring to Fvar, excess "
        "variance, and normalized excess variance "
        "respectively (see"
        " Vaughan et al. 2003 for details).",
    )
    _add_default_args(parser, ["loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    with log.log_to_file("HENexcvar.log"):
        filelist = []
        for fname in args.files:
            lcurve = load_lcurve(fname)
            if args.norm == "fvar":
                start, stop, res = lcurve.analyze_segments(
                    fvar, args.chunk_length, args.fraction_step
                )
            elif args.norm == "excvar":
                start, stop, res = lcurve.analyze_segments(
                    excvar_none, args.chunk_length, args.fraction_step
                )
            elif args.norm == "norm_excvar":
                start, stop, res = lcurve.analyze_segments(
                    excvar_norm, args.chunk_length, args.fraction_step
                )
            else:
                raise ValueError("Normalization must be fvar, norm_excvar " "or excvar")
            var, var_err = res
            out = hen_root(fname) + "_" + args.norm + ".qdp"
            save_as_qdp(
                [(start + stop) / 2, var],
                [(stop - start) / 2, var_err],
                filename=out,
            )
            filelist.append(out)

    return filelist
