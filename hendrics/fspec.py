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
from stingray.utils import assign_value_if_none

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
    HENDRICS_STAR_VALUE,
)
from stingray.lombscargle import LombScargleCrossspectrum, LombScarglePowerspectrum

from .io import sort_files, save_pds, load_data
from .io import HEN_FILE_EXTENSION, get_file_type
from .io import filter_energy


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

    all_spec = []
    for i, contents in enumerate(show_progress(fspec_iterable, total=total)):
        freq = contents.freq
        pds = contents.power
        epds = contents.power_err
        nchunks = contents.m
        if hasattr(contents, "cs_all") and contents.cs_all is not None:
            all_spec.extend(contents.cs_all)
        rebin = 1
        norm = contents.norm
        fftlen = contents.fftlen
        if i == 0:
            rebin0, norm0, freq0 = rebin, norm, freq
            tot_pds = pds * nchunks

            tot_epds = epds**2 * nchunks
            tot_npds = nchunks
            tot_contents = copy.deepcopy(contents)
        else:
            assert np.all(rebin == rebin0), "Files must be rebinned in the same way"
            np.testing.assert_array_almost_equal(
                freq,
                freq0,
                decimal=int(-np.log10(1 / fftlen) + 2),
                err_msg="Frequencies must coincide",
            )
            assert norm == norm0, "Files must have the same normalization"

            tot_pds += pds * nchunks
            tot_epds += epds**2 * nchunks
            tot_npds += nchunks

    if len(all_spec) > 0:
        tot_contents.cs_all = all_spec

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

    if hasattr(lc1, "apply_gtis"):
        lc1.apply_gtis()
        lc2.apply_gtis()

    return lc1, lc2


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
        new_ev = events.apply_mask(slice(first, last, 1))
        new_ev.gti = np.asarray([[start, end]])
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
    bintime=1,
    pdsrebin=1,
    normalization="leahy",
    back_ctrate=0.0,
    noclobber=False,
    outname=None,
    save_all=False,
    save_dyn=False,
    save_lcs=False,
    no_auxil=False,
    test=False,
    emin=None,
    emax=None,
    ignore_gti=False,
    lombscargle=False,
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
        If specified, output file name. If not specified or None, the new file
        will have the same root as the input light curve and the '_pds' suffix
    emin : float, default None
        Minimum energy of the photons
    emax : float, default None
        Maximum energy of the photons
    lombscargle : bool
        Use the Lomb-Scargle periodogram instead of AveragedPowerspectrum

    """
    root = hen_root(lcfile)
    label = ""
    if emin is not None or emax is not None:
        emin_label = f"{emin:g}" if emin is not None else HENDRICS_STAR_VALUE
        emax_label = f"{emax:g}" if emax is not None else HENDRICS_STAR_VALUE
        label += f"_{emin_label}-{emax_label}keV"
    if lombscargle:
        label += "_LS"
    if outname is None:
        outname = root + label + "_pds" + HEN_FILE_EXTENSION
    if noclobber and os.path.exists(outname):
        warnings.warn("File exists, and noclobber option used. Skipping")
        return

    ftype, data = get_file_type(lcfile)
    if ignore_gti:
        data.gti = np.asarray([[data.gti[0, 0], data.gti[-1, 1]]])

    if (emin is not None or emax is not None) and ftype != "events":
        warnings.warn("Energy selection only makes sense for event lists")
    elif ftype == "events":
        data, _ = filter_energy(data, emin, emax)

    mjdref = data.mjdref
    instr = data.instr

    if hasattr(data, "dt"):
        bintime = max(data.dt, bintime)

    if ftype != "events":
        bintime = None

    if lombscargle:
        pds = LombScarglePowerspectrum(
            data,
            dt=bintime,
            norm=normalization.lower(),
        )
        save_all = False
    else:
        pds = AveragedPowerspectrum(
            data,
            dt=bintime,
            segment_size=fftlen,
            save_all=save_dyn,
            norm=normalization.lower(),
        )

    if pdsrebin is not None and pdsrebin != 1:
        pds = pds.rebin(pdsrebin)
    pds.instr = instr
    pds.fftlen = fftlen
    pds.back_phots = back_ctrate * fftlen
    pds.mjdref = mjdref

    log.info("Saving PDS to %s" % outname)
    save_pds(
        pds,
        outname,
        save_all=save_all,
        save_dyn=save_dyn,
        save_lcs=save_lcs,
        no_auxil=no_auxil,
    )
    return outname


def calc_cpds(
    lcfile1,
    lcfile2,
    fftlen,
    bintime=1,
    pdsrebin=1,
    outname=None,
    normalization="leahy",
    back_ctrate=0.0,
    noclobber=False,
    save_all=False,
    save_dyn=False,
    save_lcs=False,
    no_auxil=False,
    test=False,
    emin=None,
    emax=None,
    ignore_gti=False,
    lombscargle=False,
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
    emin : float, default None
        Minimum energy of the photons
    emax : float, default None
        Maximum energy of the photons
    lombscargle : bool
        Use the Lomb-Scargle periodogram instead of AveragedPowerspectrum

    """
    label = ""
    if emin is not None or emax is not None:
        emin_label = f"{emin:g}" if emin is not None else HENDRICS_STAR_VALUE
        emax_label = f"{emax:g}" if emax is not None else HENDRICS_STAR_VALUE
        label += f"_{emin_label}-{emax_label}keV"
    if lombscargle:
        label += "_LS"
    if outname is None:
        root = cn if (cn := common_name(lcfile1, lcfile2)) != "" else "cpds"
        outname = root + label + "_cpds" + HEN_FILE_EXTENSION

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

    if (emin is not None or emax is not None) and (
        ftype1 != "events" or ftype2 != "events"
    ):
        warnings.warn("Energy selection only makes sense for event lists")
    if ftype1 == "events":
        lc1, _ = filter_energy(lc1, emin, emax)
    if ftype2 == "events":
        lc2, _ = filter_energy(lc2, emin, emax)

    if hasattr(lc1, "dt"):
        assert lc1.dt == lc2.dt, "Light curves are sampled differently"

    lc1, lc2 = sync_gtis(lc1, lc2)
    if ignore_gti:
        lc1.gti = lc2.gti = np.asarray([[lc1.gti[0, 0], lc1.gti[-1, 1]]])

    if lc1.mjdref != lc2.mjdref:
        lc2 = lc2.change_mjdref(lc1.mjdref)
    mjdref = lc1.mjdref

    if hasattr(lc1, "dt"):
        bintime = max(lc1.dt, bintime)

    if ftype1 != "events":
        bintime = None

    if lombscargle:
        cpds = LombScargleCrossspectrum(
            lc1,
            lc2,
            dt=bintime,
            norm=normalization.lower(),
        )
        save_all = False
    else:
        cpds = AveragedCrossspectrum(
            lc1,
            lc2,
            dt=bintime,
            segment_size=fftlen,
            save_all=save_dyn,
            norm=normalization.lower(),
        )

    if pdsrebin is not None and pdsrebin != 1:
        cpds = cpds.rebin(pdsrebin)

    cpds.instrs = instr1 + "," + instr2
    cpds.fftlen = fftlen
    cpds.back_phots = back_ctrate * fftlen
    cpds.mjdref = mjdref
    lags = cpds.time_lag()
    lags_err = np.nan
    if len(lags) == 2:
        lags, lags_err = lags
    cpds.lag = lags
    cpds.lag_err = lags_err

    log.info("Saving CPDS to %s" % outname)
    save_pds(
        cpds,
        outname,
        save_all=save_all,
        save_dyn=save_dyn,
        save_lcs=save_lcs,
        no_auxil=no_auxil,
    )
    return outname


def calc_fspec(
    files,
    fftlen,
    do_calc_pds=True,
    do_calc_cpds=True,
    do_calc_cospectrum=True,
    do_calc_lags=True,
    save_dyn=False,
    no_auxil=False,
    save_lcs=False,
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
    emin=None,
    emax=None,
    ignore_gti=False,
    lombscargle=False,
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
    emin : float, default None
        Minimum energy of the photons
    emax : float, default None
        Maximum energy of the photons
    lombscargle : bool
        Use the Lomb-Scargle periodogram instead of AveragedPowerspectrum

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
                no_auxil=no_auxil,
                save_lcs=save_lcs,
                bintime=bintime,
                pdsrebin=pdsrebin,
                normalization=normalization.lower(),
                back_ctrate=back_ctrate,
                noclobber=noclobber,
                save_all=save_all,
                test=test,
                emin=emin,
                emax=emax,
                ignore_gti=ignore_gti,
                lombscargle=lombscargle,
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
        no_auxil=no_auxil,
        save_lcs=save_lcs,
        bintime=bintime,
        pdsrebin=pdsrebin,
        normalization=normalization.lower(),
        back_ctrate=back_ctrate,
        noclobber=noclobber,
        save_all=save_all,
        test=test,
        emin=emin,
        emax=emax,
        ignore_gti=ignore_gti,
        lombscargle=lombscargle,
    )

    funcargs = []

    for i_f, f in enumerate(files1):
        f1, f2 = f, files2[i_f]

        outdir = os.path.dirname(f1)
        if outdir == "":
            outdir = os.getcwd()

        outname = None
        outr = outroot

        if len(files1) > 1 and outroot is None:
            outr = common_name(f1, f2, default="%d" % i_f)

        if outr is not None:
            outname = os.path.join(
                outdir,
                outr.replace(HEN_FILE_EXTENSION, "") + "_cpds" + HEN_FILE_EXTENSION,
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
        "--ignore-gtis",
        help="Ignore GTIs. USE AT YOUR OWN RISK",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-all",
        help=(
            "Save all information contained in spectra, including light curves "
            "and dynamical spectra."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-lcs",
        help="Save all information contained in spectra, including light curves.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--no-auxil",
        help="Do not save auxiliary spectra (e.g. pds1 and pds2 of cross spectrum)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--test",
        help="Only to be used in testing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--emin",
        default=None,
        type=float,
        help="Minimum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--emax",
        default=None,
        type=float,
        help="Maximum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--lombscargle",
        help="Use Lomb-Scargle periodogram or cross spectrum (will ignore segment_size)",
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
        bintime = interpret_bintime(args.bintime)

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
            bintime=bintime,
            pdsrebin=pdsrebin,
            outroot=args.outroot,
            normalization=normalization,
            nproc=1,
            back_ctrate=args.back,
            noclobber=args.noclobber,
            ignore_instr=args.ignore_instr,
            save_all=args.save_all,
            save_dyn=args.save_dyn,
            save_lcs=args.save_lcs,
            no_auxil=args.no_auxil,
            test=args.test,
            emin=args.emin,
            emax=args.emax,
            ignore_gti=args.ignore_gtis,
            lombscargle=args.lombscargle,
        )
