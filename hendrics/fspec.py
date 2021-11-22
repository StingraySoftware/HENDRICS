# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to calculate frequency spectra."""

import copy
import warnings
import contextlib
import os
from stingray.gti import cross_gtis
from stingray.crossspectrum import AveragedCrossspectrum
from stingray.powerspectrum import AveragedPowerspectrum
from stingray.utils import show_progress
from stingray.gti import time_intervals_from_gtis
from stingray.events import EventList
import numpy as np
from astropy import log
from astropy.logger import AstropyUserWarning
from .base import (
    hen_root,
    common_name,
    _assign_value_if_none,
    interpret_bintime,
)
from .io import sort_files, save_pds, load_data
from .io import HEN_FILE_EXTENSION, get_file_type


def average_periodograms(fspec_iterable, total=None):
    """Sum a list (or iterable) of power density spectra.

    Examples
    --------
    >>> pds = AveragedPowerspectrum()
    >>> pds.freq = np.asarray([1, 2, 3])
    >>> pds.power = np.asarray([3, 3, 3])
    >>> pds.power_err = np.asarray([0.1, 0.1, 0.1])
    >>> pds.m = 1
    >>> pds.fftlen = 128
    >>> pds1 = copy.deepcopy(pds)
    >>> pds1.m = 2
    >>> tot_pds = average_periodograms([pds, pds1])
    >>> np.allclose(tot_pds.power, pds.power)
    True
    >>> np.allclose(tot_pds.power_err, pds.power_err / np.sqrt(3))
    True
    >>> tot_pds.m
    3
    """

    for i, contents in enumerate(show_progress(fspec_iterable, total=total)):
        freq = contents.freq
        pds = contents.power
        epds = contents.power_err
        nchunks = contents.m
        rebin = 1
        norm = contents.norm
        fftlen = contents.fftlen
        if i == 0:
            rebin0, norm0, freq0 = rebin, norm, freq
            tot_pds = pds * nchunks

            tot_epds = epds ** 2 * nchunks
            tot_npds = nchunks
            tot_contents = copy.copy(contents)
        else:
            assert np.all(
                rebin == rebin0
            ), "Files must be rebinned in the same way"
            np.testing.assert_array_almost_equal(
                freq,
                freq0,
                decimal=int(-np.log10(1 / fftlen) + 2),
                err_msg="Frequencies must coincide",
            )
            assert norm == norm0, "Files must have the same normalization"

            tot_pds += pds * nchunks
            tot_epds += epds ** 2 * nchunks
            tot_npds += nchunks

    tot_contents.power = tot_pds / tot_npds
    tot_contents.power_err = np.sqrt(tot_epds) / tot_npds
    tot_contents.m = tot_npds

    return tot_contents


def _wrap_fun_cpds(arglist):
    f1, f2, outname, kwargs = arglist
    return calc_cpds(f1, f2, outname=outname, **kwargs)


def _wrap_fun_pds(argdict):
    fname = argdict["fname"]
    argdict.pop("fname")
    return calc_pds(fname, **argdict)


def sync_gtis(lc1, lc2):
    """Sync gtis between light curves or event lists.

    Has to work with new and old versions of stingray.

    Examples
    --------
    >>> from stingray.events import EventList
    >>> from stingray.lightcurve import Lightcurve
    >>> ev1 = EventList(
    ...     time=np.sort(np.random.uniform(1, 10, 3)), gti=[[1, 10]])
    >>> ev2 = EventList(time=np.sort(np.random.uniform(0, 9, 4)), gti=[[0, 9]])
    >>> e1, e2 = sync_gtis(ev1, ev2)
    >>> np.allclose(e1.gti, [[1, 9]])
    True
    >>> np.allclose(e2.gti, [[1, 9]])
    True
    >>> lc1 = Lightcurve(
    ...     time=[0.5, 1.5, 2.5], counts=[2, 2, 3], dt=1, gti=[[0, 3]])
    >>> lc2 = Lightcurve(
    ...     time=[1.5, 2.5, 3.5, 4.5], counts=[2, 2, 3, 3], dt=1, gti=[[1, 5]])
    >>> lc1._apply_gtis = lc1.apply_gtis
    >>> lc2._apply_gtis = lc2.apply_gtis
    >>> l1, l2 = sync_gtis(lc1, lc2)
    >>> np.allclose(l1.gti, [[1, 3]])
    True
    >>> np.allclose(l2.gti, [[1, 3]])
    True
    """
    gti = cross_gtis([lc1.gti, lc2.gti])
    lc1.gti = gti
    lc2.gti = gti
    if hasattr(lc1, "_apply_gtis"):
        # Compatibility with old versions of stingray
        lc1.apply_gtis = lc1._apply_gtis
        lc2.apply_gtis = lc2._apply_gtis

    if hasattr(lc1, "apply_gtis"):
        lc1.apply_gtis()
        lc2.apply_gtis()

    # compatibility with old versions of stingray
    if hasattr(lc1, "tseg") and lc1.tseg != lc2.tseg:
        lc1.tseg = np.max(lc1.gti) - np.min(lc1.gti)
        lc2.tseg = np.max(lc1.gti) - np.min(lc1.gti)
    return lc1, lc2


def _format_lc_data(data, type, fftlen=512.0, bintime=1.0):
    if type == "events":
        events = data
        gtilength = events.gti[:, 1] - events.gti[:, 0]
        events.gti = events.gti[gtilength >= fftlen]
        lc_data = list(events.to_lc_list(dt=bintime))
    else:
        lc = data
        if bintime > lc.dt:
            lcrebin = np.rint(bintime / lc.dt)
            log.info("Rebinning lcs by a factor %d" % lcrebin)
            lc = lc.rebin(bintime)
            # To fix problem with float128
            lc.counts = lc.counts.astype(float)
        lc_data = lc
    return lc_data


def _distribute_events(events, chunk_length):
    """Split event list in chunks.

    Examples
    --------
    >>> ev = EventList([1, 2, 3, 4, 5, 6], gti=[[0.5, 6.5]])
    >>> ev.pi = np.ones_like(ev.time)
    >>> ev.mjdref = 56780.
    >>> ev_lists = list(_distribute_events(ev, 2))
    >>> np.allclose(ev_lists[0].time, [1, 2])
    True
    >>> np.allclose(ev_lists[1].time, [3, 4])
    True
    >>> np.allclose(ev_lists[2].time, [5, 6])
    True
    >>> np.allclose(ev_lists[0].gti, [[0.5, 2.5]])
    True
    >>> ev_lists[0].mjdref == ev.mjdref
    True
    >>> ev_lists[2].mjdref == ev.mjdref
    True
    >>> np.allclose(ev_lists[1].pi, [1, 1])
    True
    """
    gti = events.gti
    start_times, stop_times = time_intervals_from_gtis(gti, chunk_length)
    for start, end in zip(start_times, stop_times):
        first, last = np.searchsorted(events.time, [start, end])
        new_ev = EventList(
            events.time[first:last], gti=np.asarray([[start, end]])
        )
        for attr in events.__dict__.keys():
            if attr == "gti":
                continue
            val = getattr(events, attr)
            if np.size(val) == np.size(events.time):
                val = val[first:last]
            setattr(new_ev, attr, val)
        yield new_ev


def _provide_periodograms(events, fftlen, dt, norm):
    for new_ev in _distribute_events(events, fftlen):
        # Hack: epsilon slightly below zero, to allow for a GTI to be recognized as such
        new_ev.gti[:, 1] += dt / 10
        pds = AveragedPowerspectrum(
            new_ev, dt=dt, segment_size=fftlen, norm=norm, silent=True
        )
        pds.fftlen = fftlen
        yield pds


def _provide_cross_periodograms(events1, events2, fftlen, dt, norm):
    length = events1.gti[-1, 1] - events1.gti[0, 0]
    total = int(length / fftlen)
    ev1_iter = _distribute_events(events1, fftlen)
    ev2_iter = _distribute_events(events2, fftlen)
    for new_ev in zip(ev1_iter, ev2_iter):
        new_ev1, new_ev2 = new_ev
        new_ev1.gti[:, 1] += dt / 10
        new_ev2.gti[:, 1] += dt / 10

        with contextlib.redirect_stdout(open(os.devnull, "w")):
            pds = AveragedCrossspectrum(
                new_ev1,
                new_ev2,
                dt=dt,
                segment_size=fftlen,
                norm=norm,
                silent=True,
            )
        pds.fftlen = fftlen
        yield pds


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
    save_all=False,
    test=False,
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

    ftype, data = get_file_type(lcfile)
    mjdref = data.mjdref
    instr = data.instr

    length = data.gti[-1, 1] - data.gti[0, 0]
    if hasattr(data, "dt"):
        bintime = max(data.dt, bintime)

    nbins = int(length / bintime)

    # New Stingray machinery, hopefully more efficient and consistent
    if hasattr(AveragedPowerspectrum, "from_events"):
        if ftype == "events":
            pds = AveragedPowerspectrum.from_events(
                data, dt=bintime, segment_size=fftlen
            )
        elif ftype == "lc":
            pds = AveragedPowerspectrum.from_lightcurve(
                data, segment_size=fftlen
            )
        if pds.power_err is None:
            pds.power_err = pds.power / np.sqrt(pds.m)
    else:
        if ftype == "events" and (test or nbins > 10 ** 7):
            print("Long observation. Using split analysis")
            length = data.gti[-1, 1] - data.gti[0, 0]
            total = int(length / fftlen)
            pds = average_periodograms(
                _provide_periodograms(
                    data, fftlen, bintime, norm=normalization.lower()
                ),
                total=total,
            )
        else:
            lc_data = _format_lc_data(
                data, ftype, bintime=bintime, fftlen=fftlen
            )

            pds = AveragedPowerspectrum(
                lc_data, segment_size=fftlen, norm=normalization.lower()
            )

    if pdsrebin is not None and pdsrebin != 1:
        pds = pds.rebin(pdsrebin)

    pds.instr = instr
    pds.fftlen = fftlen
    pds.back_phots = back_ctrate * fftlen
    pds.mjdref = mjdref

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
    save_all=False,
    test=False,
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
    ftype1, lc1 = get_file_type(lcfile1)
    log.info("Loading file %s..." % lcfile2)
    ftype2, lc2 = get_file_type(lcfile2)
    instr1 = lc1.instr
    instr2 = lc2.instr

    if ftype1 != ftype2:
        raise ValueError(
            "Please use similar data files for the two time "
            "series (e.g. both events or both light curves)"
        )
    if hasattr(lc1, "dt"):
        assert lc1.dt == lc2.dt, "Light curves are sampled differently"

    lc1, lc2 = sync_gtis(lc1, lc2)
    if lc1.mjdref != lc2.mjdref:
        lc2 = lc2.change_mjdref(lc1.mjdref)
    mjdref = lc1.mjdref

    length = lc1.gti[-1, 1] - lc1.gti[0, 0]
    if hasattr(lc1, "dt"):
        bintime = max(lc1.dt, bintime)

    nbins = int(length / bintime)

    # New Stingray machinery, hopefully more efficient and consistent
    if hasattr(AveragedPowerspectrum, "from_events"):
        if ftype1 == "events":
            cpds = AveragedCrossspectrum.from_events(
                lc1, lc2, dt=bintime, segment_size=fftlen
            )
        elif ftype1 == "lc":
            cpds = AveragedCrossspectrum.from_lightcurve(
                lc1, lc2, segment_size=fftlen
            )
        if cpds.power_err is None:
            cpds.power_err = np.sqrt(cpds.power) / np.sqrt(cpds.m)
    else:
        if ftype1 == "events" and (test or nbins > 10 ** 7):
            print("Long observation. Using split analysis")
            length = lc1.gti[-1, 1] - lc1.gti[0, 0]
            total = int(length / fftlen)
            cpds = average_periodograms(
                _provide_cross_periodograms(
                    lc1, lc2, fftlen, bintime, norm=normalization.lower()
                ),
                total=total,
            )
        else:
            lc1 = _format_lc_data(lc1, ftype1, fftlen=fftlen, bintime=bintime)
            lc2 = _format_lc_data(lc2, ftype2, fftlen=fftlen, bintime=bintime)

            cpds = AveragedCrossspectrum(
                lc1, lc2, segment_size=fftlen, norm=normalization.lower()
            )

    if pdsrebin is not None and pdsrebin != 1:
        cpds = cpds.rebin(pdsrebin)

    cpds.instrs = instr1 + "," + instr2
    cpds.fftlen = fftlen
    cpds.back_phots = back_ctrate * fftlen
    cpds.mjdref = mjdref
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
    save_all=False,
    test=False,
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
            wfd = dict(
                fftlen=fftlen,
                save_dyn=save_dyn,
                bintime=bintime,
                pdsrebin=pdsrebin,
                normalization=normalization.lower(),
                back_ctrate=back_ctrate,
                noclobber=noclobber,
                save_all=save_all,
                test=test,
            )
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

    argdict = dict(
        fftlen=fftlen,
        save_dyn=save_dyn,
        bintime=bintime,
        pdsrebin=pdsrebin,
        normalization=normalization.lower(),
        back_ctrate=back_ctrate,
        noclobber=noclobber,
        save_all=save_all,
        test=test,
    )

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
    parser.add_argument(
        "--test",
        help="Only to be used in testing",
        default=False,
        action="store_true",
    )
    _add_default_args(parser, ["loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

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
            test=args.test,
        )
