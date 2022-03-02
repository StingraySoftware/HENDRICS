# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform input/output operations."""

import sys
import os
import glob
import copy
import re
from typing import Tuple

from collections.abc import Iterable
import importlib
import warnings
import pickle
import os.path
import numpy as np

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
from .base import find_peaks_in_image

try:
    _ = np.complex256
    HAS_C256 = True
except Exception:
    HAS_C256 = False

cpl128 = np.dtype([(str("real"), np.double), (str("imag"), np.double)])
if HAS_C256:
    cpl256 = np.dtype(
        [(str("real"), np.longdouble), (str("imag"), np.longdouble)]
    )


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
            best_peaks, best_stat = search_best_peaks(
                self.freq, self.stat, threshold
            )
        else:
            best_cands = find_peaks_in_image(
                self.stat, n=10, threshold_abs=threshold
            )
            best_peaks = []
            best_stat = []
            for i, idx in enumerate(best_cands):
                f, fdot = self.freq[idx[0], idx[1]], self.fdots[idx[0], idx[1]]
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


def filter_energy(
    ev: EventList, emin: float, emax: float
) -> Tuple[EventList, str]:
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
        emin = np.min(energy)
    if emax is None:
        emax = np.max(energy)

    good = (energy >= emin) & (energy <= emax)
    ev.time = times[good]
    ev.energy = energy[good]
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


def get_file_extension(fname):
    """Get the file extension."""
    return os.path.splitext(fname)[1]


def get_file_format(fname):
    """Decide the file format of the file.

    Examples
    --------
    >>> get_file_format('bu.p')
    'pickle'
    >>> get_file_format('bu.nc')
    'nc'
    >>> get_file_format('bu.evt')
    'fits'
    >>> get_file_format('bu.txt')
    'text'
    >>> get_file_format('bu.pdfghj')
    Traceback (most recent call last):
        ...
    RuntimeError: File format pdfghj not recognized
    """
    ext = get_file_extension(fname)
    if ext == ".p":
        return "pickle"
    elif ext == ".nc":
        return "nc"
    elif ext in [".evt", ".fits"]:
        return "fits"
    elif ext in [".txt", ".qdp", ".csv"]:
        return "text"
    else:
        raise RuntimeError(f"File format {ext[1:]} " f"not recognized")


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
            to_save = eval(to_save.replace("__bool__", ""))
        out[k] = to_save

    rootgrp.close()

    return out


def _dum(x):
    return x


# ----- Functions to handle file types
def get_file_type(fname, raw_data=False):
    """Return the file type and its contents.

    Only works for hendrics-format pickle or netcdf files.
    """
    contents = load_data(fname)

    ftype_raw = contents["__sr__class__type__"]
    if "Lightcurve" in ftype_raw:
        ftype = "lc"
        fun = load_lcurve
    elif "Color" in ftype_raw:
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
    out = dict(
        time=eventlist.time,
        gti=eventlist.gti,
        pi=eventlist.pi,
        mjdref=eventlist.mjdref,
        tstart=np.min(eventlist.gti),
        tstop=np.max(eventlist.gti),
    )

    out["__sr__class__type__"] = str(type(eventlist))

    if hasattr(eventlist, "instr") and eventlist.instr is not None:
        out["instr"] = eventlist.instr.lower()
    else:
        out["instr"] = "unknown"
    for attr in ["energy", "cal_pi", "detector_id"]:
        if hasattr(eventlist, attr) and getattr(eventlist, attr) is not None:
            out[attr] = getattr(eventlist, attr)

    if hasattr(eventlist, "header") and eventlist.header is not None:
        out["header"] = eventlist.header
    for attr in ["mission", "ephem", "timeref", "timesys"]:
        if hasattr(eventlist, attr) and getattr(eventlist, attr) is not None:
            out[attr] = getattr(eventlist, attr).lower()

    if get_file_format(fname) == "pickle":
        _save_data_pickle(out, fname)
    elif get_file_format(fname) == "nc":
        _save_data_nc(out, fname)


def load_events(fname):
    """Load events from a file."""
    if get_file_format(fname) == "pickle":
        out = _load_data_pickle(fname)
    elif get_file_format(fname) == "nc":
        out = _load_data_nc(fname)

    eventlist = EventList()

    eventlist.time = out["time"]
    eventlist.gti = out["gti"]

    for attr in ["pi", "cal_pi", "detector_id", "energy"]:
        if attr in out:
            setattr(eventlist, attr, force_iterable(out[attr]))
    if "mjdref" in list(out.keys()):
        eventlist.mjdref = out["mjdref"]
    if "instr" in list(out.keys()):
        eventlist.instr = out["instr"].lower()
    if "header" in list(out.keys()):
        eventlist.header = out["header"]
    if "mission" in list(out.keys()):
        eventlist.mission = out["mission"].lower()
    else:
        eventlist.mission = ""

    for attr in ["mission", "ephem", "timeref", "timesys"]:
        if attr in list(out.keys()):
            setattr(eventlist, attr, out[attr].lower())
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
    out = {}

    out["__sr__class__type__"] = str(lctype)

    out["counts"] = lcurve.counts
    out["counts_err"] = lcurve.counts_err
    out["time"] = lcurve.time
    out["dt"] = lcurve.dt
    out["gti"] = lcurve.gti
    out["err_dist"] = lcurve.err_dist
    out["mjdref"] = lcurve.mjdref
    out["tstart"] = lcurve.tstart
    out["tseg"] = lcurve.tseg
    if hasattr(lcurve, "header"):
        out["header"] = lcurve.header
    if hasattr(lcurve, "expo"):
        out["expo"] = lcurve.expo
    if hasattr(lcurve, "base"):
        out["base"] = lcurve.base
    if lctype == "Color":
        out["e_intervals"] = lcurve.e_intervals
    elif hasattr(lcurve, "e_interval") and lcurve.e_interval is not None:
        out["e_interval"] = lcurve.e_interval
    if hasattr(lcurve, "use_pi"):
        out["use_pi"] = lcurve.use_pi

    if hasattr(lcurve, "instr") and lcurve.instr is not None:
        out["instr"] = lcurve.instr.lower()
    else:
        out["instr"] = "unknown"
    if hasattr(lcurve, "mission") and lcurve.mission is not None:
        out["mission"] = lcurve.mission.lower()

    if get_file_format(fname) == "pickle":
        return _save_data_pickle(out, fname)
    elif get_file_format(fname) == "nc":
        return _save_data_nc(out, fname)


def load_lcurve(fname):
    """Load light curve from a file."""
    if get_file_format(fname) == "pickle":
        data = _load_data_pickle(fname)
    elif get_file_format(fname) == "nc":
        data = _load_data_nc(fname)

    lcurve = Lightcurve(
        data["time"],
        data["counts"],
        err=data["counts_err"],
        gti=data["gti"],
        err_dist=data["err_dist"],
        mjdref=data["mjdref"],
        dt=data["dt"],
        skip_checks=True,
    )

    if "instr" in list(data.keys()) and data["instr"] is not None:
        lcurve.instr = data["instr"].lower()
    if "mission" in list(data.keys()) and data["mission"] is not None:
        lcurve.mission = data["mission"].lower()
    if "expo" in list(data.keys()):
        lcurve.expo = data["expo"]
    if "e_intervals" in list(data.keys()):
        lcurve.e_intervals = data["e_intervals"]
    if "e_interval" in list(data.keys()):
        lcurve.e_interval = data["e_interval"]
    if "use_pi" in list(data.keys()):
        lcurve.use_pi = data["use_pi"]
    if "header" in list(data.keys()):
        lcurve.header = data["header"]
    if "base" in list(data.keys()):
        lcurve.base = data["base"]

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
def save_pds(cpds, fname, save_all=False):
    """Save PDS in a file."""
    from .base import mkdir_p

    outdata = copy.copy(cpds.__dict__)
    outdata["__sr__class__type__"] = str(type(cpds))

    if not hasattr(cpds, "instr"):
        outdata["instr"] = "unknown"

    for attr in ["show_progress", "amplitude"]:
        if hasattr(cpds, attr):
            outdata[attr] = getattr(cpds, attr)

    outdir = fname.replace(HEN_FILE_EXTENSION, "")
    if save_all:
        mkdir_p(outdir)

    for attr in ["lc1", "lc2", "pds1", "pds2"]:
        if save_all and hasattr(cpds, attr):
            value = getattr(cpds, attr)

            outf = f"__{attr}__" + HEN_FILE_EXTENSION
            if "lc" in attr and isinstance(value, Lightcurve):
                save_lcurve(value, os.path.join(outdir, outf))
            elif "pds" in attr and isinstance(value, Crossspectrum):
                save_pds(value, os.path.join(outdir, outf), save_all=False)
        outdata.pop(attr, None)

    if "cs_all" in outdata:
        if save_all:
            for i, c in enumerate(cpds.cs_all):
                save_pds(
                    c,
                    os.path.join(
                        outdir, "__cs__{}__".format(i) + HEN_FILE_EXTENSION
                    ),
                )
        outdata.pop("cs_all")

    if "best_fits" in outdata and cpds.best_fits is not None:
        model_files = []
        for i, b in enumerate(cpds.best_fits):
            mfile = os.path.join(
                outdir,
                fname.replace(HEN_FILE_EXTENSION, "__mod{}__.p".format(i)),
            )
            save_model(b, mfile)
            model_files.append(mfile)
        outdata.pop("best_fits")

    if get_file_format(fname) == "pickle":
        return _save_data_pickle(outdata, fname)
    elif get_file_format(fname) == "nc":
        return _save_data_nc(outdata, fname)


def load_pds(fname, nosub=False):
    """Load PDS from a file."""
    if get_file_format(fname) == "pickle":
        data = _load_data_pickle(fname)
    elif get_file_format(fname) == "nc":
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

    outdir = fname.replace(HEN_FILE_EXTENSION, "")
    modelfiles = glob.glob(
        os.path.join(outdir, fname.replace(HEN_FILE_EXTENSION, "__mod*__.p"))
    )
    cpds.best_fits = None
    if len(modelfiles) >= 1:
        bmodels = []
        for mfile in modelfiles:
            if os.path.exists(mfile):
                bmodels.append(load_model(mfile)[0])
        cpds.best_fits = bmodels

    if nosub:
        return cpds

    lc1_name = os.path.join(outdir, "__lc1__" + HEN_FILE_EXTENSION)
    lc2_name = os.path.join(outdir, "__lc2__" + HEN_FILE_EXTENSION)
    pds1_name = os.path.join(outdir, "__pds1__" + HEN_FILE_EXTENSION)
    pds2_name = os.path.join(outdir, "__pds2__" + HEN_FILE_EXTENSION)
    cs_all_names = glob.glob(
        os.path.join(outdir, "__cs__[0-9]__" + HEN_FILE_EXTENSION)
    )

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
                raise ValueError(
                    contents[kind_key] + ": unrecognized kind string"
                )

            log10_part = contents[log10_key]
            if isinstance(contents[integer_key], Iterable):
                integer_part = np.array(contents[integer_key], dtype=dtype)
                float_part = np.array(contents[float_key], dtype=dtype)
            else:
                integer_part = dtype(contents[integer_key])
                float_part = dtype(contents[float_key])

            contents[kcorr] = (integer_part + float_part) * 10.0 ** log10_part

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
        dum = np.min(np.abs(var))
        if dum < 1 and dum > 0.0:
            var_log10 = np.floor(np.log10(dum))

        var = np.asarray(var) / (10.0 ** var_log10)
        var_I = np.floor(var).astype(int)
        var_F = np.array(var - var_I, dtype=np.double)
    else:
        if np.abs(var) < 1 and np.abs(var) > 0.0:
            var_log10 = np.floor(np.log10(np.abs(var)))

        var = np.asarray(var) / 10.0 ** var_log10
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
    if get_file_format(fname) == "pickle":
        _save_data_pickle(struct, fname)
    elif get_file_format(fname) == "nc":
        _save_data_nc(struct, fname)


def load_data(fname):
    """Load generic data in hendrics format."""
    if get_file_format(fname) == "pickle":
        return _load_data_pickle(fname)
    elif get_file_format(fname) == "nc":
        return _load_data_nc(fname)
    else:
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
    print(
        "Instrument:    {0}/{1}\n".format(
            info["Telescope"], info["Instrument"]
        )
    )
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
        ftype, contents = get_file_type(fname, raw_data=True)
        print("This file contains:", end="\n\n")
        mjdref = 0.0 * u.d
        if "mjdref" in contents:
            mjdref = Time(contents["mjdref"], format="mjd")

        tstart = None
        tstop = None
        tseg = None

        for k in sorted(contents.keys()):
            if k == "header" and not args.print_header:
                continue
            if k == "tstart":
                timeval = contents[k] * u.s
                val = f"MET {contents[k]} s (MJD {mjdref + timeval.to(u.d)})"
                tstart = timeval
            elif k == "tstop":
                timeval = contents[k] * u.s
                val = f"MET {contents[k]} s (MJD {mjdref + timeval.to(u.d)})"
                tstop = timeval
            elif k == "tseg":
                val = f"{contents[k]} s"
                tseg = contents[k] * u.s
            else:
                val = contents[k]
            if isinstance(val, Iterable) and not is_string(val):
                length = len(val)
                if len(val) < 4:
                    val = repr(list(val[:4]))
                else:
                    val = repr(list(val[:4])).replace("]", "") + "...]"
                    val = "{} (len {})".format(val, length)
            print((k + ":").ljust(15), val, end="\n\n")

        if tseg is None and (tstart is not None and tstop is not None):
            print(("length:").ljust(15), tstop - tstart, end="\n\n")

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
                "Accepted callable models have only one "
                "non-keyword argument"
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
                "Accepted callable models have only one "
                "non-keyword argument"
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
