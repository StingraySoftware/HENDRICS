#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:55:47 2017

@author: marta
"""
from __future__ import print_function, division
from .io import load_lcurve
from .io import save_as_qdp
from .base import hen_root
import logging
from stingray.utils import excess_variance


def fvar(lc):
    return excess_variance(lc, normalization='fvar')


def excvar(lc):
    return excess_variance(lc)    


def main(args=None):
    import argparse
    description = ('Calculate excess variance in light curve chunks')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument('-c', "--chunk-length", type=float, default= 20,
                        help="Length in seconds of the light curve chunks")
    parser.add_argument("--fraction-step", type=float, default= 0.5,
                        help="If the step is not a full chunk_length but less,"
                        "this indicates the ratio between step step and"
                        " `chunk_length`")
    parser.add_argument("--norm", type=str, default="excvar",
                        help="Choose between fvar and excvar normalization")
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

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENlags.log', level=numeric_level,
                        filemode='w')
    filelist=[]
    for fname in args.files:
        lcurve = load_lcurve(fname)
        if args.norm == "fvar":
            start, stop, res = lcurve.analyze_lc_chunks(args.chunk_length,fvar, 
                                                        args.fraction_step)
        elif args.norm == "excvar":
            start, stop, res = lcurve.analyze_lc_chunks(args.chunk_length,excvar, 
                                                        args.fraction_step) 
        else:
            raise ValueError("Normalization must be fvar or excvar")
        var, var_err = res
        out = hen_root(fname) + "_" + args.norm + '.qdp'
        save_as_qdp([(start+stop)/2, var],[(stop-start)/2, var_err],
                    filename=out)
        filelist.append(out)
    
    return filelist
        
        