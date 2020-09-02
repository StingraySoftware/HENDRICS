# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to calculate frequency spectra."""

import warnings
import os
from stingray.gti import cross_gtis
from stingray.crossspectrum import AveragedCrossspectrum
from stingray.powerspectrum import AveragedPowerspectrum
import numpy as np
from astropy import log
from astropy.logger import AstropyUserWarning
from .base import (
    hen_root,
    common_name,
    _assign_value_if_none,
    interpret_bintime,
)
from .io import sort_files, save_pds, load_lcurve
from .io import HEN_FILE_EXTENSION


def _wrap_fun_cpds(arglist):
    f1, f2, outname, kwargs = arglist
    return calc_cpds(f1, f2, outname=outname, **kwargs)


def _wrap_fun_pds(argdict):
    fname = argdict["fname"]
    argdict.pop("fname")
    return calc_pds(fname, **argdict)


def sync_gtis(lc1, lc2):
    """Sync gtis between light curves.

    Has to work with new and old versions of stingray."""
    gti = cross_gtis([lc1.gti, lc2.gti])
    lc1.gti = gti
    lc2.gti = gti
    if hasattr(lc1, "_apply_gtis"):
        # Compatibility with old versions of stingray
        lc1.apply_gtis = lc1._apply_gtis
        lc2.apply_gtis = lc2._apply_gtis
    lc1.apply_gtis()
    lc2.apply_gtis()
    # compatibility with old versions of stingray
    if lc1.tseg != lc2.tseg:  # pragma: no cover
        lc1.tseg = np.max(lc1.gti) - np.min(lc1.gti)
        lc2.tseg = np.max(lc1.gti) - np.min(lc1.gti)
    return lc1, lc2


def calc_pds(
    lcfile,
    fftlen,
    save_dyn=False,
    bintime=1,
    pdsrebin=1,
    normalization="leahy",
    back_ctrate=0.0,
    noclobber=False,
    outname=None,
    save_all=True,
):
    """Calculate the PDS from an input light curve file.

    Parameters
    ----------
    lcfile : str
        The light curve file
    fftlen : float
        The length of the chunks over which FFTs will be calculated, in seconds

    Other Parameters
    ----------------
    save_dyn : bool
        If True, save the dynamical power spectrum
    bintime : float
        The bin time. If different from that of the light curve, a rebinning is
        performed
    pdsrebin : int
        Rebin the PDS of this factor.
    normalization: str
        'Leahy', 'frac', 'rms', or any normalization accepted by ``stingray``.
        Default 'Leahy'
    back_ctrate : float
        The non-source count rate
    noclobber : bool
        If True, do not overwrite existing files
    outname : str
        If speficied, output file name. If not specified or None, the new file
        will have the same root as the input light curve and the '_pds' suffix
    """
    root = hen_root(lcfile)
    if outname is None:
        outname = root + "_pds" + HEN_FILE_EXTENSION
    if noclobber and os.path.exists(outname):
        warnings.warn("File exists, and noclobber option used. Skipping")
        return

    log.info("Loading file %s..." % lcfile)
    lc = load_lcurve(lcfile)
    instr = lc.instr

    if bintime > lc.dt:
        lcrebin = np.rint(bintime / lc.dt)
        log.info("Rebinning lcs by a factor %d" % lcrebin)
        lc = lc.rebin(bintime)
        # To fix problem with float128
        lc.counts = lc.counts.astype(float)
        lc.instr = instr

    pds = AveragedPowerspectrum(
        lc, segment_size=fftlen, norm=normalization.lower()
    )

    if pdsrebin is not None and pdsrebin != 1:
        pds = pds.rebin(pdsrebin)

    pds.instr = instr
    pds.fftlen = fftlen
    pds.back_phots = back_ctrate * fftlen
    pds.mjdref = lc.mjdref

    log.info("Saving PDS to %s" % outname)
    save_pds(pds, outname, save_all=save_all)
    return outname


def calc_cpds(
    lcfile1,
    lcfile2,
    fftlen,
    save_dyn=False,
    bintime=1,
    pdsrebin=1,
    outname="cpds" + HEN_FILE_EXTENSION,
    normalization="leahy",
    back_ctrate=0.0,
    noclobber=False,
    save_all=True,
):
    """Calculate the CPDS from a pair of input light curve files.

    Parameters
    ----------
    lcfile1 : str
        The first light curve file
    lcfile2 : str
        The second light curve file
    fftlen : float
        The length of the chunks over which FFTs will be calculated, in seconds

    Other Parameters
    ----------------
    save_dyn : bool
        If True, save the dynamical power spectrum
    bintime : float
        The bin time. If different from that of the light curve, a rebinning is
        performed
    pdsrebin : int
        Rebin the PDS of this factor.
    normalization : str
        'Leahy', 'frac', 'rms', or any normalization accepted by ``stingray``.
        Default 'Leahy'
    back_ctrate : float
        The non-source count rate
    noclobber : bool
        If True, do not overwrite existing files
    outname : str
        Output file name for the cpds. Default: cpds.[nc|p]
    """
    if noclobber and os.path.exists(outname):
        warnings.warn("File exists, and noclobber option used. Skipping")
        return

    log.info("Loading file %s..." % lcfile1)
    lc1 = load_lcurve(lcfile1)
    log.info("Loading file %s..." % lcfile2)
    lc2 = load_lcurve(lcfile2)
    instr1 = lc1.instr
    instr2 = lc2.instr

    assert lc1.dt == lc2.dt, "Light curves are sampled differently"
    dt = lc1.dt

    lc1, lc2 = sync_gtis(lc1, lc2)

    if bintime > dt:
        lcrebin = np.rint(bintime / dt)
        log.info("Rebinning lcs by a factor %d" % lcrebin)
        lc1 = lc1.rebin(bintime)
        # To fix problem with float128
        lc1.counts = lc1.counts.astype(float)
        lc1.instr = instr1
        lc2 = lc2.rebin(bintime)
        lc2.counts = lc2.counts.astype(float)
        lc2.instr = instr2

    if lc1.mjdref != lc2.mjdref:
        lc2 = lc2.change_mjdref(lc1.mjdref)

    cpds = AveragedCrossspectrum(
        lc1, lc2, segment_size=fftlen, norm=normalization.lower()
    )

    if pdsrebin is not None and pdsrebin != 1:
        cpds = cpds.rebin(pdsrebin)

    cpds.instrs = instr1 + "," + instr2
    cpds.fftlen = fftlen
    cpds.back_phots = back_ctrate * fftlen
    cpds.mjdref = lc1.mjdref
    lags, lags_err = cpds.time_lag()
    cpds.lag = lags
    cpds.lag_err = lags

    log.info("Saving CPDS to %s" % outname)
    save_pds(cpds, outname, save_all=save_all)
    return outname


def calc_fspec(
    files,
    fftlen,
    do_calc_pds=True,
    do_calc_cpds=True,
    do_calc_cospectrum=True,
    do_calc_lags=True,
    save_dyn=False,
    bintime=1,
    pdsrebin=1,
    outroot=None,
    normalization="leahy",
    nproc=1,
    back_ctrate=0.0,
    noclobber=False,
    ignore_instr=False,
    save_all=True,
):
    r"""Calculate the frequency spectra: the PDS, the cospectrum, ...

    Parameters
    ----------
    files : list of str
        List of input file names
    fftlen : float
        length of chunks to perform the FFT on.

    Other Parameters
    ----------------
    save_dyn : bool
        If True, save the dynamical power spectrum
    bintime : float
        The bin time. If different from that of the light curve, a rebinning is
        performed
    pdsrebin : int
        Rebin the PDS of this factor.
    normalization : str
        'Leahy' [3] or 'rms' [4] [5]. Default 'Leahy'.
    back_ctrate : float
        The non-source count rate
    noclobber : bool
        If True, do not overwrite existing files
    outroot : str
        Output file name root
    nproc : int
        Number of processors to use to parallelize the processing of multiple
        files
    ignore_instr : bool
        Ignore instruments; files are alternated in the two channels

    References
    ----------
    [3] Leahy et al. 1983, ApJ, 266, 160.

    [4] Belloni & Hasinger 1990, A&A, 230, 103

    [5] Miyamoto et al. 1991, ApJ, 383, 784

    """

    log.info("Using %s normalization" % normalization)
    log.info("Using %s processors" % nproc)

    if do_calc_pds:
        wrapped_file_dicts = []
        for f in files:
            wfd = {
                "fftlen": fftlen,
                "save_dyn": save_dyn,
                "bintime": bintime,
                "pdsrebin": pdsrebin,
                "normalization": normalization.lower(),
                "back_ctrate": back_ctrate,
                "noclobber": noclobber,
                "save_all": save_all,
            }
            wfd["fname"] = f
            wrapped_file_dicts.append(wfd)

        [_wrap_fun_pds(w) for w in wrapped_file_dicts]

    if not do_calc_cpds or len(files) < 2:
        return

    if ignore_instr:
        files1 = files[0::2]
        files2 = files[1::2]
    else:
        log.info("Sorting file list")
        sorted_files = sort_files(files)

        warnings.warn(
            "Beware! For cpds and derivatives, I assume that the "
            "files are from only two instruments and in pairs "
            "(even in random order)"
        )

        instrs = list(sorted_files.keys())

        files1 = sorted_files[instrs[0]]
        files2 = sorted_files[instrs[1]]

    assert len(files1) == len(files2), "An even number of files is needed"

    argdict = {
        "fftlen": fftlen,
        "save_dyn": save_dyn,
        "bintime": bintime,
        "pdsrebin": pdsrebin,
        "normalization": normalization.lower(),
        "back_ctrate": back_ctrate,
        "noclobber": noclobber,
        "save_all": save_all,
    }

    funcargs = []

    for i_f, f in enumerate(files1):
        f1, f2 = f, files2[i_f]

        outdir = os.path.dirname(f1)
        if outdir == "":
            outdir = os.getcwd()

        outr = _assign_value_if_none(
            outroot, common_name(f1, f2, default="%d" % i_f)
        )

        outname = os.path.join(
            outdir,
            outr.replace(HEN_FILE_EXTENSION, "")
            + "_cpds"
            + HEN_FILE_EXTENSION,
        )

        funcargs.append([f1, f2, outname, argdict])

    [_wrap_fun_cpds(fa) for fa in funcargs]


def _normalize(array, ref=0):
    """Normalize array in terms of standard deviation.

    Examples
    --------
    >>> n = 10000
    >>> array1 = np.random.normal(0, 1, n)
    >>> array2 = np.random.normal(0, 1, n)
    >>> array = array1 ** 2 + array2 ** 2
    >>> newarr = _normalize(array)
    >>> np.isclose(np.std(newarr), 1, atol=0.0001)
    True
    """
    m = ref
    std = np.std(array)
    newarr = np.zeros_like(array)
    good = array > m
    newarr[good] = (array[good] - ref) / std
    return newarr


def dumpdyn(fname, plot=False):
    raise NotImplementedError(
        "Dynamical power spectrum is being refactored. "
        "Sorry for the inconvenience. In the meantime, "
        "you can load the data into Stingray using "
        "`cs = hendrics.io.load_pds(fname)` and find "
        "the dynamical PDS/CPDS in cs.cs_all"
    )


def dumpdyn_main(args=None):
    """Main function called by the `HENdumpdyn` command line script."""
    import argparse

    description = (
        "Dump dynamical (cross) power spectra. "
        "This script is being reimplemented. Please be "
        "patient :)"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "files",
        help=("List of files in any valid HENDRICS " "format for PDS or CPDS"),
        nargs="+",
    )
    parser.add_argument(
        "--noplot", help="plot results", default=False, action="store_true"
    )

    args = parser.parse_args(args)

    fnames = args.files

    for f in fnames:
        dumpdyn(f, plot=not args.noplot)


def main(args=None):
    """Main function called by the `HENfspec` command line script."""
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Create frequency spectra (PDS, CPDS, cospectrum) "
        "starting from well-defined input ligthcurves"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of light curve files", nargs="+")
    parser.add_argument(
        "-b",
        "--bintime",
        type=float,
        default=1 / 4096,
        help="Light curve bin time; if negative, interpreted"
        + " as negative power of 2."
        + " Default: 2^-10, or keep input lc bin time"
        + " (whatever is larger)",
    )
    parser.add_argument(
        "-r",
        "--rebin",
        type=int,
        default=1,
        help="(C)PDS rebinning to apply. Default: none",
    )
    parser.add_argument(
        "-f",
        "--fftlen",
        type=float,
        default=512,
        help="Length of FFTs. Default: 512 s",
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        default="PDS,CPDS,cos",
        help="Spectra to calculate, as comma-separated list"
        + " (Accepted: PDS and CPDS;"
        + ' Default: "PDS,CPDS")',
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="leahy",
        help="Normalization to use"
        + " (Accepted: leahy and rms;"
        + ' Default: "leahy")',
    )
    parser.add_argument(
        "--noclobber",
        help="Do not overwrite existing files",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--outroot",
        type=str,
        default=None,
        help="Root of output file names for CPDS only",
    )
    parser.add_argument(
        "--back",
        help=("Estimated background (non-source) count rate"),
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--save-dyn",
        help="save dynamical power spectrum",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ignore-instr",
        help="Ignore instrument names in channels",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-all",
        help="Save all information contained in spectra,"
        " including single pdss and light curves.",
        default=False,
        action="store_true",
    )
    _add_default_args(parser, ["loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)
    log.info("Starting")

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    with log.log_to_file("HENfspec.log"):
        bintime = np.longdouble(interpret_bintime(args.bintime))

        fftlen = args.fftlen
        pdsrebin = args.rebin
        normalization = args.norm
        if normalization.lower() not in [
            "frac",
            "abs",
            "leahy",
            "none",
            "rms",
        ]:
            warnings.warn("Beware! Unknown normalization!", AstropyUserWarning)
            normalization = "leahy"
        if normalization == "rms":
            normalization = "frac"

        do_cpds = do_pds = do_cos = do_lag = False
        kinds = args.kind.split(",")
        for k in kinds:
            if k == "PDS":
                do_pds = True
            elif k == "CPDS":
                do_cpds = True
            elif k == "cos" or k == "cospectrum":
                do_cos = True
                do_cpds = True
            elif k == "lag":
                do_lag = True
                do_cpds = True

        calc_fspec(
            args.files,
            fftlen,
            do_calc_pds=do_pds,
            do_calc_cpds=do_cpds,
            do_calc_cospectrum=do_cos,
            do_calc_lags=do_lag,
            save_dyn=args.save_dyn,
            bintime=bintime,
            pdsrebin=pdsrebin,
            outroot=args.outroot,
            normalization=normalization,
            nproc=1,
            back_ctrate=args.back,
            noclobber=args.noclobber,
            ignore_instr=args.ignore_instr,
            save_all=args.save_all,
        )
