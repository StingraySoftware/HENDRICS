# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform input/output operations."""

import sys
import shutil
import os
import glob
import copy
import re
from typing import Tuple
import logging

from collections.abc import Iterable
import importlib
import warnings
import pickle
import os.path
import numpy as np
from astropy.table import Table

from hendrics.base import get_file_extension, get_file_format, splitext_improved
from stingray.base import StingrayObject, StingrayTimeseries

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import netCDF4 as nc

    HEN_FILE_EXTENSION = ".nc"
    HAS_NETCDF = True
except ImportError:
    msg = "Warning! NetCDF is not available. Using pickle format."
    warnings.warn(msg)
    HEN_FILE_EXTENSION = ".p"
    HAS_NETCDF = False
    pass
from astropy.modeling.core import Model
from astropy import log
from astropy.logger import AstropyUserWarning
from astropy.io import fits
from stingray.utils import assign_value_if_none
from stingray.events import EventList
from stingray.lightcurve import Lightcurve
from stingray.powerspectrum import Powerspectrum, AveragedPowerspectrum
from stingray.crossspectrum import Crossspectrum, AveragedCrossspectrum
from stingray.pulse.modeling import SincSquareModel
from stingray.pulse.search import search_best_peaks

from .base import _order_list_of_arrays, _empty, is_string, force_iterable
from .base import find_peaks_in_image, hen_root

try:
    _ = np.complex256
    HAS_C256 = True
except Exception:
    HAS_C256 = False

cpl128 = np.dtype([(str("real"), np.double), (str("imag"), np.double)])
if HAS_C256:
    cpl256 = np.dtype([(str("real"), np.longdouble), (str("imag"), np.longdouble)])


class EFPeriodogram(object):
    def __init__(
        self,
        freq=None,
        stat=None,
        kind=None,
        nbin=None,
        N=None,
        oversample=None,
        M=None,
        pepoch=None,
        mjdref=None,
        peaks=None,
        peak_stat=None,
        best_fits=None,
        fdots=0,
        fddots=0,
        segment_size=1e32,
        filename="",
        parfile=None,
        emin=None,
        emax=None,
        ncounts=None,
        upperlim=None,
    ):
        self.freq = freq
        self.stat = stat
        self.kind = kind
        self.nbin = nbin
        self.oversample = oversample
        self.N = N
        self.peaks = peaks
        self.peak_stat = peak_stat
        self.best_fits = best_fits
        self.fdots = fdots
        self.fddots = fddots
        self.M = M
        self.segment_size = segment_size
        self.filename = filename
        self.parfile = parfile
        self.emin = emin
        self.emax = emax
        self.pepoch = pepoch
        self.mjdref = mjdref
        self.upperlim = upperlim
        self.ncounts = ncounts

    def find_peaks(self, conflevel=99.0):
        from .base import z2_n_detection_level, fold_detection_level

        ntrial = self.stat.size
        if hasattr(self, "oversample") and self.oversample is not None:
            ntrial /= self.oversample
            ntrial = int(ntrial)

        epsilon = 1 - conflevel / 100
        if self.kind == "Z2n":
            threshold = z2_n_detection_level(
                epsilon=epsilon,
                n=self.N,
                ntrial=ntrial,
                n_summed_spectra=int(self.M),
            )
        else:
            threshold = fold_detection_level(
                nbin=int(self.nbin), epsilon=epsilon, ntrial=ntrial
            )

        if len(self.stat.shape) == 1:
            best_peaks, best_stat = search_best_peaks(self.freq, self.stat, threshold)
        else:
            best_cands = find_peaks_in_image(self.stat, n=10, threshold_abs=threshold)
            best_peaks = []
            best_stat = []
            for i, idx in enumerate(best_cands):
                f, fdot = (
                    self.freq[idx[0], idx[1]],
                    self.fdots[idx[0], idx[1]],
                )
                best_peaks.append([f, fdot])
                best_stat.append(self.stat[idx[0], idx[1]])
        best_peaks = np.asarray(best_peaks)
        best_stat = np.asarray(best_stat)
        if len(best_peaks) > 0:
            self.peaks = best_peaks
            self.peak_stat = best_stat
        return best_peaks, best_stat


def get_energy_from_events(ev):
    if hasattr(ev, "energy") and ev.energy is not None:
        energy = ev.energy
        elabel = "Energy"
    elif hasattr(ev, "pi") and ev.pi is not None:
        energy = ev.pi
        elabel = "PI"
        ev.energy = energy
    else:
        energy = np.ones_like(ev.time)
        elabel = ""
    return elabel, energy


def filter_energy(ev: EventList, emin: float, emax: float) -> Tuple[EventList, str]:
    """Filter event list by energy (or PI)

    If an ``energy`` attribute is present, uses it. Otherwise, it switches
    automatically to ``pi``

    Examples
    --------
    >>> import doctest
    >>> from contextlib import redirect_stderr
    >>> import sys
    >>> time = np.arange(5)
    >>> energy = np.array([0, 0, 30, 4, 1])
    >>> events = EventList(time=time, energy=energy)
    >>> ev_out, elabel = filter_energy(events, 3, None)
    >>> np.all(ev_out.time == [2, 3])
    True
    >>> elabel == 'Energy'
    True
    >>> events = EventList(time=time, pi=energy)
    >>> with warnings.catch_warnings(record=True) as w:
    ...     ev_out, elabel = filter_energy(events, None, 20)  # doctest: +ELLIPSIS
    >>> "No energy information in event list" in str(w[-1].message)
    True
    >>> np.all(ev_out.time == [0, 1, 3, 4])
    True
    >>> elabel == 'PI'
    True
    >>> events = EventList(time=time, pi=energy)
    >>> ev_out, elabel = filter_energy(events, None, None)  # doctest: +ELLIPSIS
    >>> np.all(ev_out.time == time)
    True
    >>> elabel == 'PI'
    True
    >>> events = EventList(time=time)
    >>> with redirect_stderr(sys.stdout):
    ...     ev_out, elabel = filter_energy(events, 3, None)  # doctest: +ELLIPSIS
    ERROR:...No Energy or PI...
    >>> np.all(ev_out.time == time)
    True
    >>> elabel == ''
    True
    """
    times = ev.time

    elabel, energy = get_energy_from_events(ev)
    # For some reason the doctest doesn't work if I don't do this instead
    # of using warnings.warn
    if elabel == "":
        log.error(
            "No Energy or PI information available. "
            "No energy filter applied to events"
        )
        return ev, ""

    if emax is None and emin is None:
        return ev, elabel
    # For some reason the doctest doesn't work if I don't do this instead
    # of using warnings.warn
    if elabel.lower() == "pi" and (emax is not None or emin is not None):
        warnings.warn(
            f"No energy information in event list "
            f"while filtering between {emin} and {emax}. "
            f"Definition of events.energy is now based on PI."
        )
    if emin is None:
        emin = np.min(energy) - 1
    if emax is None:
        emax = np.max(energy) + 1

    good = (energy >= emin) & (energy <= emax)
    ev.apply_mask(good, inplace=True)
    # ev.time = times[good]
    # ev.energy = energy[good]
    return ev, elabel


def _get_key(dict_like, key):
    """
    Examples
    --------
    >>> a = dict(b=1)
    >>> _get_key(a, 'b')
    1
    >>> _get_key(a, 'c') == ""
     True
    """
    try:
        return dict_like[key]
    except KeyError:
        return ""


def high_precision_keyword_read(hdr, keyword):
    """Read FITS header keywords, also if split in two.

    In the case where the keyword is split in two, like

        MJDREF = MJDREFI + MJDREFF

    in some missions, this function returns the summed value. Otherwise, the
    content of the single keyword

    Parameters
    ----------
    hdr : dict_like
        The header structure, or a dictionary
    keyword : str
        The key to read in the header

    Returns
    -------
    value : long double
        The value of the key, or None if keyword not present

    Examples
    --------
    >>> hdr = dict(keywordS=1.25)
    >>> high_precision_keyword_read(hdr, 'keywordS')
    1.25
    >>> hdr = dict(keywordI=1, keywordF=0.25)
    >>> high_precision_keyword_read(hdr, 'keywordS')
    1.25
    >>> high_precision_keyword_read(hdr, 'bubabuab') is None
    True
    """
    if keyword in hdr:
        return np.longdouble(hdr[keyword])

    if len(keyword) == 8:
        keyword = keyword[:7]

    if keyword + "I" in hdr and keyword + "F" in hdr:
        value_i = np.longdouble(hdr[keyword + "I"])
        value_f = np.longdouble(hdr[keyword + "F"])
        return value_i + value_f
    else:
        return None


def read_header_key(fits_file, key, hdu=1):
    """Read the header key ``key`` from HDU ``hdu`` of a fits file.

    Parameters
    ----------
    fits_file: str
    key: str
        The keyword to be read

    Other Parameters
    ----------------
    hdu : int
    """
    from astropy.io import fits as pf

    hdulist = pf.open(fits_file)
    try:
        value = hdulist[hdu].header[key]
    except KeyError:  # pragma: no cover
        value = ""
    hdulist.close()
    return value


def ref_mjd(fits_file, hdu=1):
    """Read MJDREFF+ MJDREFI or, if failed, MJDREF, from the FITS header.

    Parameters
    ----------
    fits_file : str

    Returns
    -------
    mjdref : numpy.longdouble
        the reference MJD

    Other Parameters
    ----------------
    hdu : int
    """
    from astropy.io import fits as pf

    if isinstance(fits_file, Iterable) and not is_string(fits_file):
        fits_file = fits_file[0]
        log.info("opening %s", fits_file)
    with pf.open(fits_file) as hdul:
        return high_precision_keyword_read(hdul[hdu].header, "MJDREF")


# ---- Base function to save NetCDF4 files
def save_as_netcdf(vars, varnames, formats, fname):
    """Save variables in a NetCDF4 file."""
    rootgrp = nc.Dataset(fname, "w", format="NETCDF4")

    for iv, v in enumerate(vars):
        dims = {}
        dimname = varnames[iv] + "dim"
        dimspec = (varnames[iv] + "dim",)

        if formats[iv] == "c32":
            # Too complicated. Let's decrease precision
            warnings.warn("complex256 yet unsupported", AstropyUserWarning)
            formats[iv] = "c16"

        if formats[iv] == "c16":
            v = np.asarray(v)
            # unicode_literals breaks something, I need to specify str.
            if "cpl128" not in rootgrp.cmptypes.keys():
                complex128_t = rootgrp.createCompoundType(cpl128, "cpl128")
            vcomp = np.empty(v.shape, dtype=cpl128)
            vcomp["real"] = v.real.astype(np.float64)
            vcomp["imag"] = v.imag.astype(np.float64)
            v = vcomp
            formats[iv] = complex128_t

        unsized = False
        try:
            len(v)
        except TypeError:
            unsized = True

        if isinstance(v, Iterable) and formats[iv] != str and not unsized:
            dim = len(v)
            dims[dimname] = dim

            if isinstance(v[0], Iterable):
                dim = len(v[0])
                dims[dimname + "_2"] = dim
                dimspec = (dimname, dimname + "_2")
        else:
            dims[dimname] = 1

        for dimname in dims.keys():
            rootgrp.createDimension(dimname, dims[dimname])
        vnc = rootgrp.createVariable(varnames[iv], formats[iv], dimspec)
        try:
            if formats[iv] == str:
                vnc[0] = v
            else:
                vnc[:] = v
        except Exception:
            log.error("Bad variable:", varnames[iv], formats[iv], dimspec, v)
            raise
    rootgrp.close()


def read_from_netcdf(fname):
    """Read from a netCDF4 file."""
    rootgrp = nc.Dataset(fname)
    out = {}
    for k in rootgrp.variables.keys():
        dum = rootgrp.variables[k]
        values = dum.__array__()
        # Handle special case of complex
        if dum.dtype == cpl128:
            arr = np.empty(values.shape, dtype=np.complex128)
            arr.real = values[str("real")]
            arr.imag = values[str("imag")]
            values = arr

        # Handle special case of complex
        if HAS_C256 and dum.dtype == cpl256:
            arr = np.empty(values.shape, dtype=np.complex256)
            arr.real = values[str("real")]
            arr.imag = values[str("imag")]
            values = arr

        if dum.dtype == str or dum.size == 1:
            to_save = values[0]
        else:
            to_save = values
        if isinstance(to_save, (str, bytes)) and to_save.startswith("__bool_"):
            # Boolean single value
            to_save = eval(to_save.replace("__bool__", ""))
        # Boolean array
        elif k.startswith("__bool__"):
            to_save = to_save.astype(bool)
            k = k.replace("__bool__", "")

        out[k] = to_save

    rootgrp.close()

    return out


def _dum(x):
    return x


def recognize_stingray_table(obj):
    """

    Examples
    --------
    >>> obj = AveragedCrossspectrum()
    >>> obj.freq = np.arange(10)
    >>> obj.power = np.random.random(10)
    >>> recognize_stingray_table(obj.to_astropy_table())
    'AveragedPowerspectrum'
    >>> obj.pds1 = obj.power
    >>> recognize_stingray_table(obj.to_astropy_table())
    'AveragedCrossspectrum'
    >>> obj = EventList(np.arange(10))
    >>> recognize_stingray_table(obj.to_astropy_table())
    'EventList'
    >>> obj = Lightcurve(np.arange(10), np.arange(10))
    >>> recognize_stingray_table(obj.to_astropy_table())
    'Lightcurve'
    >>> obj = Table()
    >>> recognize_stingray_table(obj)
    Traceback (most recent call last):
    ...
    ValueError: Object not recognized...
    """
    if "hue" in obj.colnames:
        return "Powercolors"
    if "power" in obj.colnames:
        if np.iscomplex(obj["power"][1]) or "pds1" in obj.colnames:
            return "AveragedCrossspectrum"
        return "AveragedPowerspectrum"
    if "counts" in obj.colnames:
        return "Lightcurve"
    if "time" in obj.colnames:
        return "EventList"
    raise ValueError(f"Object not recognized:\n{obj}")


# ----- Functions to handle file types
def get_file_type(fname, raw_data=False):
    """Return the file type and its contents.

    Only works for hendrics-format pickle or netcdf files,
    or stingray outputs.
    """
    contents_raw = load_data(fname)
    if isinstance(contents_raw, Table):
        ftype_raw = recognize_stingray_table(contents_raw)
        if raw_data:
            contents = dict([(col, contents_raw[col]) for col in contents_raw.colnames])
            contents.update(contents_raw.meta)
    else:
        ftype_raw = contents_raw["__sr__class__type__"]
        contents = contents_raw

    if "Lightcurve" in ftype_raw:
        ftype = "lc"
        fun = load_lcurve
    elif ("Powercolor" in ftype_raw) or (
        "StingrayTimeseries" in ftype_raw and "hue" in contents
    ):
        ftype = "powercolor"
        fun = load_timeseries
    elif "StingrayTimeseries" in ftype_raw or "Color" in ftype_raw:
        ftype = "color"
        fun = load_lcurve
    elif "EventList" in ftype_raw:
        ftype = "events"
        fun = load_events
    elif "Crossspectrum" in ftype_raw:
        ftype = "cpds"
        fun = load_pds
    elif "Powerspectrum" in ftype_raw:
        ftype = "pds"
        fun = load_pds
    elif "gti" in ftype_raw:
        ftype = "gti"
        fun = _dum
    elif "EFPeriodogram" in ftype_raw:
        ftype = "folding"
        fun = load_folding
    else:
        raise ValueError("File format not understood")

    if not raw_data:
        contents = fun(fname)

    return ftype, contents


# ----- functions to save and load EVENT data
def save_events(eventlist, fname):
    """Save events in a file.

    Parameters
    ----------
    eventlist: :class:`stingray.EventList` object
        Event list to be saved
    fname: str
        Name of output file
    """
    save_data(eventlist, fname)


def save_timeseries(timeseries, fname):
    """Save a time series in a file.

    Parameters
    ----------
    timeseries: :class:`stingray.EventList` object
        Event list to be saved
    fname: str
        Name of output file
    """
    save_data(timeseries, fname)


def load_events(fname):
    """Load events from a file."""
    fmt = get_file_format(fname)
    if fmt == "pickle":
        out = _load_data_pickle(fname)
    elif fmt == "nc":
        out = _load_data_nc(fname)
    else:
        # Try one of the known files from Astropy
        return EventList.read(fname, fmt=fmt)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized keywords:.*")
        eventlist = EventList(**out)
    for key in out.keys():
        if hasattr(eventlist, key) and getattr(eventlist, key) is not None:
            continue
        setattr(eventlist, key, out[key])
    for attr in ["mission", "instr"]:
        if attr not in list(out.keys()):
            setattr(eventlist, attr, "")
    return eventlist


def load_timeseries(fname):
    """Load events from a file."""
    fmt = get_file_format(fname)
    if fmt == "pickle":
        out = _load_data_pickle(fname)
    elif fmt == "nc":
        out = _load_data_nc(fname)
    else:
        # Try one of the known files from Astropy
        return StingrayTimeseries.read(fname, fmt=fmt)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized keywords:.*")
        eventlist = StingrayTimeseries(**out)
    # for key in out.keys():
    #     if hasattr(eventlist, key) and getattr(eventlist, key) is not None:
    #         continue
    #     setattr(eventlist, key, out[key])
    # for attr in ["mission", "instr"]:
    #     if attr not in list(out.keys()):
    #         setattr(eventlist, attr, "")
    return eventlist


# ----- functions to save and load LCURVE data
def save_lcurve(lcurve, fname, lctype="Lightcurve"):
    """Save Light curve to file

    Parameters
    ----------
    lcurve: :class:`stingray.Lightcurve` object
        Event list to be saved
    fname: str
        Name of output file
    """

    fmt = get_file_format(fname)

    if hasattr(lcurve, "_mask") and lcurve._mask is not None and np.any(~lcurve._mask):
        logging.info("The light curve has a mask. Applying it before saving.")
        lcurve = lcurve.apply_mask(lcurve._mask, inplace=False)
        lcurve._mask = None

    if fmt not in ["nc", "pickle"]:
        return lcurve.write(fname)

    lcdict = lcurve.dict()
    lcdict["__sr__class__type__"] = str(lctype)
    save_data(lcdict, fname)


def load_lcurve(fname):
    """Load light curve from a file."""
    fmt = get_file_format(fname)
    if fmt == "pickle":
        data = _load_data_pickle(fname)
    elif fmt == "nc":
        data = _load_data_nc(fname)
    else:
        # Try one of the known files from Lightcurve
        return Lightcurve.read(fname, fmt=fmt, skip_checks=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized keywords:.*")
        time = data["time"]
        data.pop("time")

        lcurve = Lightcurve()
        lcurve.time = time

        for key in data.keys():
            vals = data[key]
            if key == "mask":
                key = "_mask"
            setattr(lcurve, key, vals)

    if "mission" not in list(data.keys()):
        lcurve.mission = ""

    return lcurve


# ---- Functions to save epoch folding results
def save_folding(efperiodogram, fname):
    """Save PDS in a file."""

    outdata = copy.copy(efperiodogram.__dict__)
    outdata["__sr__class__type__"] = "EFPeriodogram"
    if "best_fits" in outdata and efperiodogram.best_fits is not None:
        model_files = []
        for i, b in enumerate(efperiodogram.best_fits):
            mfile = fname.replace(HEN_FILE_EXTENSION, "__mod{}__.p".format(i))
            save_model(b, mfile)
            model_files.append(mfile)
        outdata.pop("best_fits")

    if get_file_format(fname) == "pickle":
        return _save_data_pickle(outdata, fname)
    elif get_file_format(fname) == "nc":
        return _save_data_nc(outdata, fname)


def load_folding(fname):
    """Load PDS from a file."""
    if get_file_format(fname) == "pickle":
        data = _load_data_pickle(fname)
    elif get_file_format(fname) == "nc":
        data = _load_data_nc(fname)

    data.pop("__sr__class__type__")

    ef = EFPeriodogram()

    for key in data.keys():
        setattr(ef, key, data[key])
    modelfiles = glob.glob(fname.replace(HEN_FILE_EXTENSION, "__mod*__.p"))
    if len(modelfiles) >= 1:
        bmodels = []
        for mfile in modelfiles:
            if os.path.exists(mfile):
                bmodels.append(load_model(mfile)[0])
        ef.best_fits = bmodels
    if ef.peaks is not None and len(np.asarray(ef.peaks).shape) == 0:
        ef.peaks = [ef.peaks]
    return ef


# ---- Functions to save PDSs
def save_pds(
    cpds, fname, save_all=False, save_dyn=False, no_auxil=False, save_lcs=False
):
    """Save PDS in a file."""
    from .base import mkdir_p

    if os.path.exists(fname):
        os.unlink(fname)
    cpds = copy.deepcopy(cpds)

    if save_all:
        save_dyn = True
        no_auxil = False
        save_lcs = True

    basename, ext = splitext_improved(fname)
    outdir = basename

    if save_dyn or not no_auxil or save_lcs:
        mkdir_p(outdir)

    fmt = get_file_format(fname)

    if hasattr(cpds, "subcs"):
        del cpds.subcs
    if hasattr(cpds, "unnorm_subcs"):
        del cpds.unnorm_subcs

    if no_auxil:
        for attr in ["pds1", "pds2"]:
            if hasattr(cpds, attr):
                delattr(cpds, attr)
    for attr in ["pds1", "pds2"]:
        if hasattr(cpds, attr):
            value = getattr(cpds, attr)

            outf = f"__{attr}__" + ext
            if "pds" in attr and isinstance(value, Crossspectrum):
                outfile = os.path.join(outdir, outf)
                save_pds(value, outfile, no_auxil=True)
        if hasattr(cpds, attr):
            delattr(cpds, attr)

    for lcattr in ("lc1", "lc2"):
        if hasattr(cpds, lcattr) and save_lcs:
            lc_name = os.path.join(outdir, f"__{lcattr}__" + ext)
            lc = getattr(cpds, lcattr)
            if isinstance(lc, Iterable):
                if len(lc) > 1:
                    warnings.warn(
                        "Saving multiple light curves is not supported. Saving only one"
                    )
                lc = lc[0]
            if isinstance(lc, Lightcurve):
                save_lcurve(lc, lc_name)
                delattr(cpds, lcattr)

    for attr in ["cs_all", "unnorm_cs_all"]:
        if not hasattr(cpds, attr):
            continue
        if not save_dyn:
            delattr(cpds, attr)
            continue

        saved_outside = False
        for i, c in enumerate(getattr(cpds, attr)):
            label = attr.replace("_all", "")
            if not hasattr(c, "freq"):
                break
            save_pds(
                c,
                os.path.join(outdir, f"__{label}__{i}__" + ext),
                no_auxil=True,
            )
            saved_outside = True
        if saved_outside:
            delattr(cpds, attr)

    if hasattr(cpds, "lc1"):
        del cpds.lc1
    if hasattr(cpds, "lc2"):
        del cpds.lc2

    if not hasattr(cpds, "instr"):
        cpds.instr = "unknown"

    if hasattr(cpds, "best_fits") and cpds.best_fits is not None:
        model_files = []
        for i, b in enumerate(cpds.best_fits):
            mfile = os.path.join(
                outdir,
                basename + "__mod{}__.p".format(i),
            )
            save_model(b, mfile)
            model_files.append(mfile)
        del cpds.best_fits

    if fmt not in ["nc", "pickle"]:
        return cpds.write(fname, fmt=fmt)

    outdata = copy.copy(cpds.__dict__)
    outdata["__sr__class__type__"] = str(type(cpds))

    if fmt == "pickle":
        return _save_data_pickle(outdata, fname)
    elif fmt == "nc":
        return _save_data_nc(outdata, fname)


def remove_pds(fname):
    """Remove the pds file and the directory with auxiliary information."""
    outdir, _ = splitext_improved(fname)
    modelfiles = glob.glob(
        os.path.join(outdir, fname.replace(HEN_FILE_EXTENSION, "__mod*__.p"))
    )
    for mfile in modelfiles:
        os.unlink(mfile)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.unlink(fname)


def load_pds(fname, nosub=False):
    """Load PDS from a file."""
    rootname, ext = splitext_improved(fname)
    fmt = get_file_format(fname)

    if fmt not in ["pickle", "nc"]:
        dummy = Table.read(fname, format=fmt)
        if "pds1" in dummy.colnames or "power.real" in dummy.colnames:
            cpds = AveragedCrossspectrum.read(fname, fmt=fmt)
        else:
            cpds = AveragedPowerspectrum.read(fname, fmt=fmt)

    else:
        if fmt == "pickle":
            data = _load_data_pickle(fname)
        elif fmt == "nc":
            data = _load_data_nc(fname)

        type_string = data["__sr__class__type__"]
        if "AveragedPowerspectrum" in type_string:
            cpds = AveragedPowerspectrum()
        elif "Powerspectrum" in type_string:
            cpds = Powerspectrum()
        elif "AveragedCrossspectrum" in type_string:
            cpds = AveragedCrossspectrum()
        elif "Crossspectrum" in type_string:
            cpds = Crossspectrum()
        else:
            raise ValueError("Unrecognized data type in file")

        data.pop("__sr__class__type__")

        for key in data.keys():
            setattr(cpds, key, data[key])

    outdir = rootname
    modelfiles = glob.glob(os.path.join(outdir, rootname + "__mod*__.p"))
    cpds.best_fits = None
    if len(modelfiles) >= 1:
        bmodels = []
        for mfile in modelfiles:
            if os.path.exists(mfile):
                bmodels.append(load_model(mfile)[0])
        cpds.best_fits = bmodels

    if nosub:
        return cpds

    lc1_name = os.path.join(outdir, "__lc1__" + ext)
    lc2_name = os.path.join(outdir, "__lc2__" + ext)
    pds1_name = os.path.join(outdir, "__pds1__" + ext)
    pds2_name = os.path.join(outdir, "__pds2__" + ext)
    cs_all_names = glob.glob(os.path.join(outdir, "__cs__[0-9]*__" + ext))
    unnorm_cs_all_names = glob.glob(os.path.join(outdir, "__unnorm_cs__[0-9]*__" + ext))

    if os.path.exists(lc1_name):
        cpds.lc1 = load_lcurve(lc1_name)
    if os.path.exists(lc2_name):
        cpds.lc2 = load_lcurve(lc2_name)
    if os.path.exists(pds1_name):
        cpds.pds1 = load_pds(pds1_name)
    if os.path.exists(pds2_name):
        cpds.pds2 = load_pds(pds2_name)
    if len(cs_all_names) > 0:
        cs_all = []
        for c in sorted(cs_all_names):
            cs_all.append(load_pds(c))
        cpds.cs_all = cs_all
    if len(unnorm_cs_all_names) > 0:
        unnorm_cs_all = []
        for c in sorted(unnorm_cs_all_names):
            unnorm_cs_all.append(load_pds(c))
        cpds.unnorm_cs_all = unnorm_cs_all

    return cpds


# ---- GENERIC function to save stuff.
def _load_data_pickle(fname, kind="data"):
    """Load generic data in pickle format."""
    log.info("Loading %s and info from %s" % (kind, fname))
    with open(fname, "rb") as fobj:
        result = pickle.load(fobj)
    return result


def _save_data_pickle(struct, fname, kind="data"):
    """Save generic data in pickle format."""
    log.info("Saving %s and info to %s" % (kind, fname))
    with open(fname, "wb") as fobj:
        pickle.dump(struct, fobj)

    return


def _load_data_nc(fname):
    """Load generic data in netcdf format."""
    contents = read_from_netcdf(fname)
    keys = list(contents.keys())

    keys_to_delete = []
    for k in keys:
        if k in keys_to_delete:
            continue

        if str(contents[k]) == str("__hen__None__type__"):
            contents[k] = None

        if k[-2:] in ["_I", "_L", "_F", "_k"]:
            kcorr = k[:-2]

            integer_key = kcorr + "_I"
            float_key = kcorr + "_F"
            kind_key = kcorr + "_k"
            log10_key = kcorr + "_L"

            if not (integer_key in keys and float_key in keys):
                continue
            # Maintain compatibility with old-style files:
            if not (kind_key in keys and log10_key in keys):
                contents[kind_key] = "longdouble"
                contents[log10_key] = 0

            keys_to_delete.extend([integer_key, float_key])
            keys_to_delete.extend([kind_key, log10_key])

            if contents[kind_key] == "longdouble":
                dtype = np.longdouble
            elif contents[kind_key] == "double":
                dtype = np.double
            else:
                raise ValueError(contents[kind_key] + ": unrecognized kind string")

            log10_part = contents[log10_key]
            if isinstance(contents[integer_key], Iterable):
                integer_part = np.array(contents[integer_key], dtype=dtype)
                float_part = np.array(contents[float_key], dtype=dtype)
            else:
                integer_part = dtype(contents[integer_key])
                float_part = dtype(contents[float_key])

            contents[kcorr] = (integer_part + float_part) * 10.0**log10_part

    for k in keys_to_delete:
        del contents[k]

    return contents


def _split_high_precision_number(varname, var, probesize):
    var_log10 = 0
    if probesize == 8:
        kind_str = "double"
    if probesize == 16:
        kind_str = "longdouble"
    if isinstance(var, Iterable):
        var = np.asarray(var)
        bad = np.isnan(var)
        dum = np.min(np.abs(var[~bad]))
        if dum < 1 and dum > 0.0:
            var_log10 = np.floor(np.log10(dum))

        var = np.asarray(var) / (10.0**var_log10)
        var[bad] = 0
        var_I = np.floor(var).astype(int)
        var_F = np.array(var - var_I, dtype=np.double)
        var_F[bad] = np.nan
    else:
        if np.abs(var) < 1 and np.abs(var) > 0.0:
            var_log10 = np.floor(np.log10(np.abs(var)))

        if np.isnan(var):
            var_I = np.asarray(0).astype(int)
            var_F = np.asarray(np.nan)
        else:
            var = np.asarray(var) / 10.0**var_log10
            var_I = int(np.floor(var))
            var_F = np.double(var - var_I)
    return var_I, var_F, var_log10, kind_str


def _save_data_nc(struct, fname, kind="data"):
    """Save generic data in netcdf format."""
    log.info("Saving %s and info to %s" % (kind, fname))
    varnames = []
    values = []
    formats = []

    for k in struct.keys():
        var = struct[k]
        if isinstance(var, bool):
            var = f"__bool__{var}"
        probe = var

        if isinstance(var, Iterable) and len(var) >= 1:
            probe = var[0]

        if is_string(var):
            probekind = str
            probesize = -1
        elif var is None:
            probekind = None
        else:
            probekind = np.result_type(probe).kind
            probesize = np.result_type(probe).itemsize

        if probekind == "f" and probesize >= 8:
            # If a (long)double, split it in integer + floating part.
            # If the number is below zero, also use a logarithm of 10 before
            # that, so that we don't lose precision
            var_I, var_F, var_log10, kind_str = _split_high_precision_number(
                k, var, probesize
            )
            values.extend([var_I, var_log10, var_F, kind_str])
            formats.extend(["i8", "i8", "f8", str])
            varnames.extend([k + "_I", k + "_L", k + "_F", k + "_k"])
        elif probekind == str:
            values.append(var)
            formats.append(probekind)
            varnames.append(k)
        elif probekind == "b":
            values.append(var.astype("u1"))
            formats.append("u1")
            varnames.append("__bool__" + k)
        elif probekind is None:
            values.append("__hen__None__type__")
            formats.append(str)
            varnames.append(k)
        else:
            values.append(var)
            formats.append(probekind + "%d" % probesize)
            varnames.append(k)

    save_as_netcdf(values, varnames, formats, fname)


def save_data(struct, fname, ftype="data"):
    """Save generic data in hendrics format."""
    fmt = get_file_format(fname)
    has_write_method = hasattr(struct, "write")
    struct_dict = struct
    if isinstance(struct, StingrayObject):
        struct_dict = struct.dict()

    if fmt in ["pickle", "nc"]:
        if "__sr__class__type__" not in struct_dict:
            struct_dict["__sr__class__type__"] = str(type(struct))
        if fmt == "pickle":
            return _save_data_pickle(struct_dict, fname, kind=ftype)
        elif fmt == "nc":
            return _save_data_nc(struct_dict, fname, kind=ftype)

    if not has_write_method:
        raise ValueError("Unrecognized data format or file format")

    struct.write(fname)


def load_data(fname):
    """Load generic data in hendrics format."""
    fmt = get_file_format(fname)
    if fmt == "pickle":
        return _load_data_pickle(fname)
    elif fmt == "nc":
        return _load_data_nc(fname)

    try:
        return Table.read(fname, format=fmt)
    except Exception as e:
        raise TypeError(
            "The file type is not recognized. Did you convert the"
            " original files into HENDRICS format (e.g. with "
            "HENreadevents or HENlcurve)?"
        )


# QDP format is often used in FTOOLS
def save_as_qdp(arrays, errors=None, filename="out.qdp", mode="w"):
    """Save arrays in a QDP file.

    Saves an array of variables, and possibly their errors, to a QDP file.

    Parameters
    ----------
    arrays: [array1, array2]
        List of variables. All variables must be arrays and of the same length.
    errors: [array1, array2]
        List of errors. The order has to be the same of arrays; the value can
        be:
        - None if no error is assigned
        - an array of same length of variable for symmetric errors
        - an array of len-2 lists for non-symmetric errors (e.g.
        [[errm1, errp1], [errm2, errp2], [errm3, errp3], ...])

    Other parameters
    ----------------
    mode : str
        the file access mode, to be passed to the open() function. Can be 'w'
        or 'a'
    """
    import numpy as np

    errors = assign_value_if_none(errors, [None for i in arrays])

    data_to_write = []
    list_of_errs = []
    for ia, ar in enumerate(arrays):
        data_to_write.append(ar)
        if errors[ia] is None:
            continue
        shape = np.shape(errors[ia])
        assert shape[0] == len(ar), "Errors and arrays must have same length"
        if len(shape) == 1:
            list_of_errs.append([ia, "S"])
            data_to_write.append(errors[ia])
        elif shape[1] == 2:
            list_of_errs.append([ia, "T"])
            mine = [k[0] for k in errors[ia]]
            maxe = [k[1] for k in errors[ia]]
            data_to_write.append(mine)
            data_to_write.append(maxe)

    print_header = True
    if os.path.exists(filename) and mode == "a":
        print_header = False
    outfile = open(filename, mode)
    if print_header:
        for lerr in list_of_errs:
            i, kind = lerr
            print("READ %s" % kind + "ERR %d" % (i + 1), file=outfile)

    length = len(data_to_write[0])
    for i in range(length):
        for idw, d in enumerate(data_to_write):
            print(d[i], file=outfile, end=" ")
        print("", file=outfile)

    outfile.close()


def save_as_ascii(cols, filename="out.txt", colnames=None, append=False):
    """Save arrays as TXT file with respective errors."""
    import numpy as np

    shape = np.shape(cols)
    ndim = len(shape)

    if ndim == 1:
        cols = [cols]
    elif ndim >= 3 or ndim == 0:
        log.error("Only one- or two-dim arrays accepted")
        return -1
    lcol = len(cols[0])

    log.debug("%s %s" % (repr(cols), repr(np.shape(cols))))
    if append:
        txtfile = open(filename, "a")
    else:
        txtfile = open(filename, "w")

    if colnames is not None:
        print("#", file=txtfile, end=" ")
        for i_c, c in enumerate(cols):
            print(colnames[i_c], file=txtfile, end=" ")
        print("", file=txtfile)
    for i in range(lcol):
        for c in cols:
            print(c[i], file=txtfile, end=" ")

        print("", file=txtfile)
    txtfile.close()
    return 0


def print_fits_info(fits_file, hdu=1):
    """Print general info about an observation."""
    from astropy.io import fits as pf
    from astropy.units import Unit
    from astropy.time import Time

    lchdulist = pf.open(fits_file)

    datahdu = lchdulist[hdu]
    header = datahdu.header

    info = {}
    info["N. events"] = _get_key(header, "NAXIS2")
    info["Telescope"] = _get_key(header, "TELESCOP")
    info["Instrument"] = _get_key(header, "INSTRUME")
    info["OBS_ID"] = _get_key(header, "OBS_ID")
    info["Target"] = _get_key(header, "OBJECT")
    info["Start"] = _get_key(header, "DATE-OBS")
    info["Stop"] = _get_key(header, "DATE-END")

    # Give time in MJD
    mjdref = high_precision_keyword_read(header, "MJDREF")
    tstart = high_precision_keyword_read(header, "TSTART")
    tstop = high_precision_keyword_read(header, "TSTOP")
    tunit = _get_key(header, "TIMEUNIT")
    start_mjd = Time(mjdref, format="mjd") + tstart * Unit(tunit)
    stop_mjd = Time(mjdref, format="mjd") + tstop * Unit(tunit)

    print("ObsID:         {0}\n".format(info["OBS_ID"]))
    print("Date:          {0} -- {1}\n".format(info["Start"], info["Stop"]))
    print("Date (MJD):    {0} -- {1}\n".format(start_mjd, stop_mjd))
    print("Instrument:    {0}/{1}\n".format(info["Telescope"], info["Instrument"]))
    print("Target:        {0}\n".format(info["Target"]))
    print("N. Events:     {0}\n".format(info["N. events"]))

    lchdulist.close()
    return info


def main(args=None):
    """Main function called by the `HENreadfile` command line script."""
    from astropy.time import Time
    import astropy.units as u
    import argparse

    description = "Print the content of HENDRICS files"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "--print-header",
        help="Print the full FITS header if present in the " "meta data.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args(args)

    for fname in args.files:
        print()
        print("-" * len(fname))
        print("{0}".format(fname))
        print("-" * len(fname))
        if fname.endswith(".fits") or fname.endswith(".evt"):
            print("This FITS file contains:", end="\n\n")
            print_fits_info(fname)
            print("-" * len(fname))
            continue
        ftype, contents = get_file_type(fname, raw_data=False)
        print(contents)

        print("-" * len(fname))


def sort_files(files):
    """Sort a list of HENDRICS files, looking at `Tstart` in each."""
    allfiles = {}
    ftypes = []

    for f in files:
        log.info("Loading file " + f)
        ftype, contents = get_file_type(f)
        instr = contents.instr
        ftypes.append(ftype)
        if instr not in list(allfiles.keys()):
            allfiles[instr] = []
        # Add file name to the dictionary
        contents.__sort__filename__ = f
        allfiles[instr].append(contents)

    # Check if files are all of the same kind (lcs, PDSs, ...)
    ftypes = list(set(ftypes))
    assert len(ftypes) == 1, "Files are not all of the same kind."

    instrs = list(allfiles.keys())
    for instr in instrs:
        contents = list(allfiles[instr])
        tstarts = [np.min(c.gti) for c in contents]
        fnames = [c.__sort__filename__ for c in contents]

        fnames = [x for (y, x) in sorted(zip(tstarts, fnames))]

        # Substitute dictionaries with the sorted list of files
        allfiles[instr] = fnames

    return allfiles


def save_model(model, fname="model.p", constraints=None):
    """Save best-fit models to data.

    Parameters
    ----------
    model : func or `astropy.modeling.core.Model` object
        The model to be saved
    fname : str, default 'models.p'
        The output file name

    Other parameters
    ----------------
    constraints: dict
        Additional model constraints. Ignored for astropy models.
    """
    modeldata = {"model": model, "constraints": None}
    if isinstance(model, (Model, SincSquareModel)):
        modeldata["kind"] = "Astropy"
    elif callable(model):
        nargs = model.__code__.co_argcount
        nkwargs = len(model.__defaults__)
        if not nargs - nkwargs == 1:
            raise TypeError(
                "Accepted callable models have only one " "non-keyword argument"
            )
        modeldata["kind"] = "callable"
        modeldata["constraints"] = constraints
    else:
        raise TypeError(
            "The model has to be an Astropy model or a callable"
            " with only one non-keyword argument"
        )

    with open(fname, "wb") as fobj:
        pickle.dump(modeldata, fobj)


def load_model(modelstring):
    if not is_string(modelstring):
        raise TypeError("modelstring has to be an existing file name")
    if not os.path.exists(modelstring):
        raise FileNotFoundError("Model file not found")

    # modelstring is a pickle file
    if modelstring.endswith(".p"):
        log.debug("Loading model from pickle file")
        with open(modelstring, "rb") as fobj:
            modeldata = pickle.load(fobj)
        return modeldata["model"], modeldata["kind"], modeldata["constraints"]
    # modelstring is a python file
    elif modelstring.endswith(".py"):
        log.debug("Loading model from Python source")
        modulename = modelstring.replace(".py", "")
        sys.path.append(os.getcwd())
        # If a module with the same name was already imported, unload it!
        # This is because the user might be using the same file name but
        # different models inside, just like we do in test_io.py
        if modulename in sys.modules:
            del sys.modules[modulename]

        # This invalidate_caches() is called to account for the case when
        # the model file does not exist the first time we call
        # importlib.import_module(). In this case, the second time we call it,
        # even if the file exists it will not exist for importlib.
        importlib.invalidate_caches()

        _model = importlib.import_module(modulename)
        model = _model.model
        constraints = None
        if hasattr(_model, "constraints"):
            constraints = _model.constraints
    else:
        raise TypeError("Unknown file type")

    if isinstance(model, Model):
        return model, "Astropy", constraints
    elif callable(model):
        nargs = model.__code__.co_argcount
        nkwargs = len(model.__defaults__)
        if not nargs - nkwargs == 1:
            raise TypeError(
                "Accepted callable models have only one " "non-keyword argument"
            )
        return model, "callable", constraints


def find_file_in_allowed_paths(fname, other_paths=None):
    """Check if file exists at its own relative/absolute path, or elsewhere.

    Parameters
    ----------
    fname : str
        The name of the file, with or without a path.

    Other Parameters
    ----------------
    other_paths : list of str
        list of other possible paths
    """
    if fname is None:
        return False
    existance_condition = os.path.exists(fname)
    if existance_condition:
        return fname
    bname = os.path.basename(fname)

    if other_paths is not None:
        for p in other_paths:
            fullpath = os.path.join(p, bname)
            if os.path.exists(fullpath):
                log.info(f"Parfile found at different path: {fullpath}")
                return fullpath

    return False


def main_filter_events(args=None):
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Filter events"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="Input event files", type=str, nargs="+")

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

    _add_default_args(
        parser,
        [
            "loglevel",
            "debug",
            "test",
        ],
    )

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    for fname in args.files:
        events = load_events(fname)
        events, _ = filter_energy(events, args.emin, args.emax)

        save_events(
            events,
            hen_root(fname) + f"_{args.emin:g}-{args.emax:g}keV" + HEN_FILE_EXTENSION,
        )
