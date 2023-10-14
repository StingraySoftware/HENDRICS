# -*- coding: utf-8 -*-
"""
@author: marta
"""

import warnings
from astropy import log
from astropy.table import Table
from astropy.logger import AstropyUserWarning
import numpy as np

try:
    from stingray.varenergyspectrum import VarEnergySpectrum as VES
    from stingray.varenergyspectrum import (
        LagSpectrum,
        RmsSpectrum,
        CovarianceSpectrum,
        CountSpectrum,
        _decode_energy_specification,
    )
except ImportError:
    VES = object
    warnings.warn("Please update stingray to the latest version.")


from .base import hen_root, interpret_bintime
from .io import load_events
from .io import save_as_qdp


def varenergy_to_astropy_table(spectrum):
    start_energy = np.asarray(spectrum.energy_intervals)[:, 0]
    stop_energy = np.asarray(spectrum.energy_intervals)[:, 1]
    res = Table(
        {
            "start_energy": start_energy,
            "stop_energy": stop_energy,
            "spectrum": spectrum.spectrum,
            "error": spectrum.spectrum_error,
        }
    )

    for attr in [
        "ref_band",
        "energy_intervals",
        "freq_interval",
        "bin_time",
        "use_pi",
        "segment_size",
        "norm",
        "return_complex",
        "norm",
    ]:
        if hasattr(spectrum, attr):
            res.meta[attr] = getattr(spectrum, attr)

    return res


def varenergy_from_astropy_table(fname):
    """

    Examples
    --------
    >>> varen = VarEnergySpectrum()
    >>> varen.spectrum = [3, 4]
    >>> varen.energy_intervals = [[3, 5], [5, 7]]
    >>> varen.spectrum_error = [0.3, 0.4]
    >>> varen.ref_band = [0, 3]
    >>> table = varenergy_to_astropy_table(varen)
    >>> spec = varenergy_from_astropy_table(table)
    >>> np.allclose(spec.spectrum, varen.spectrum)
    True
    >>> np.allclose(spec.spectrum_error, varen.spectrum_error)
    True
    >>> np.allclose(spec.energy_intervals, varen.energy_intervals)
    True
    >>> np.allclose(spec.ref_band, varen.ref_band)
    True
    >>> table.write("varenergyboubou.ecsv", overwrite=True)
    >>> spec_file = varenergy_from_astropy_table("varenergyboubou.ecsv")
    >>> np.allclose(spec.spectrum, spec_file.spectrum)
    True
    >>> np.allclose(spec.spectrum_error, spec_file.spectrum_error)
    True
    """
    if isinstance(fname, Table):
        data = fname
    else:
        data = Table.read(fname)
    varenergy = VarEnergySpectrum()

    for attr in [
        "ref_band",
        "freq_interval",
        "bin_time",
        "use_pi",
        "segment_size",
        "norm",
        "return_complex",
        "norm",
    ]:
        if attr in data.meta:
            setattr(varenergy, attr, data.meta[attr])

    varenergy.energy_intervals = list(zip(data["start_energy"], data["stop_energy"]))
    varenergy.spectrum = data["spectrum"]
    varenergy.spectrum_error = data["error"]
    return varenergy


class VarEnergySpectrum(VES):
    def __init__(self):
        for attr in [
            "ref_band",
            "freq_interval",
            "bin_time",
            "use_pi",
            "segment_size",
            "norm",
            "return_complex",
        ]:
            setattr(self, attr, None)

        for attr in ["energy_intervals", "spectrum", "spectrum_error"]:
            setattr(self, attr, None)

    def _spectrum_function(self):  # pragma: no cover
        pass


def main(args=None):
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Calculates variability-energy spectra"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-f",
        "--freq-interval",
        nargs=2,
        type=float,
        default=[0.0, 100],
        help="Frequence interval",
    )
    parser.add_argument(
        "--energy-values",
        nargs=4,
        type=str,
        default="0.3 12 5 lin".split(" "),
        help="Choose Emin, Emax, number of intervals," "interval spacing, lin or log",
    )
    parser.add_argument(
        "--segment-size",
        type=float,
        default=512,
        help="Length of the light curve intervals to be " "averaged",
    )
    parser.add_argument(
        "--ref-band",
        nargs=2,
        type=float,
        default=None,
        help="Reference band when relevant",
    )
    parser.add_argument(
        "--rms", default=False, action="store_true", help="Calculate rms"
    )
    parser.add_argument(
        "--covariance",
        default=False,
        action="store_true",
        help="Calculate covariance spectrum",
    )
    parser.add_argument(
        "--use-pi",
        default=False,
        action="store_true",
        help="Energy intervals are specified as PI channels",
    )
    parser.add_argument(
        "--cross-instr",
        default=False,
        action="store_true",
        help="Use data files in pairs, for example with the"
        "reference band from one and the subbands from "
        "the  other (useful in NuSTAR and "
        "multiple-detector missions)",
    )
    parser.add_argument(
        "--lag",
        default=False,
        action="store_true",
        help="Calculate lag-energy",
    )
    parser.add_argument(
        "--count",
        default=False,
        action="store_true",
        help="Calculate lag-energy",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Additional label to be added to file names",
    )
    parser.add_argument(
        "--norm",
        default="abs",
        help=(
            "When relevant, the normalization of the spectrum. One of "
            "['abs', 'frac', 'rms', 'leahy', 'none']"
        ),
    )
    parser.add_argument(
        "--format",
        default="ecsv",
        help=(
            "Output format for the table. Can be ECSV, QDP, or any other "
            "format accepted by astropy"
        ),
    )

    _add_default_args(parser, ["bintime", "loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)
    args.bintime = np.longdouble(interpret_bintime(args.bintime))

    if args.debug:
        args.loglevel = "DEBUG"

    label = ""
    if args.label is not None:
        label = "_" + args.label.lstrip("_")
    log.setLevel(args.loglevel)
    with log.log_to_file("HENvarenergy.log"):
        filelist = []
        energy_spec = (
            float(args.energy_values[0]),
            float(args.energy_values[1]),
            int(args.energy_values[2]),
            args.energy_values[3],
        )
        from .io import sort_files

        if args.cross_instr:
            log.info("Sorting file list")
            sorted_files = sort_files(args.files)

            warnings.warn(
                "Beware! For cpds and derivatives, I assume that the "
                "files are from only two instruments and in pairs "
                "(even in random order)"
            )

            instrs = list(sorted_files.keys())

            files1 = sorted_files[instrs[0]]
            files2 = sorted_files[instrs[1]]
        else:
            files1 = args.files
            files2 = [None] * len(args.files)

        for fnames in zip(files1, files2):
            fname = fnames[0]
            fname2 = fnames[1]

            events = load_events(fname)
            events2 = None
            if fname2 is not None:
                events2 = load_events(fname2)
            if not args.use_pi and (
                events.energy is None
                or (events2 is not None and events2.energy is None)
            ):
                raise ValueError(
                    "If --use-pi is not specified, event lists must "
                    "be calibrated! Please use HENcalibrate."
                )

            additional_output_args = {}
            if args.format == "qdp":
                additional_output_args["err_specs"] = {"serr": [3]}
            if args.format in ["hdf5"]:
                additional_output_args["serialize_meta"] = True

            if args.rms:
                rms = RmsSpectrum(
                    events,
                    freq_interval=args.freq_interval,
                    energy_spec=energy_spec,
                    segment_size=args.segment_size,
                    bin_time=args.bintime,
                    events2=events2,
                    use_pi=args.use_pi,
                    norm=args.norm,
                )
                outfile = hen_root(fname) + label + "_rms." + args.format
                out_table = varenergy_to_astropy_table(rms)
                out_table.write(outfile, overwrite=True, **additional_output_args)
                filelist.append(outfile)

            if args.lag:
                lag = LagSpectrum(
                    events,
                    freq_interval=args.freq_interval,
                    energy_spec=energy_spec,
                    ref_band=args.ref_band,
                    segment_size=args.segment_size,
                    bin_time=args.bintime,
                    events2=events2,
                    use_pi=args.use_pi,
                )
                outfile = hen_root(fname) + label + "_lag." + args.format
                out_table = varenergy_to_astropy_table(lag)
                out_table.write(outfile, overwrite=True, **additional_output_args)
                filelist.append(outfile)

            if args.count:
                cts = CountSpectrum(
                    events,
                    energy_spec=energy_spec,
                    use_pi=args.use_pi,
                )
                outfile = hen_root(fname) + label + "_count." + args.format
                out_table = varenergy_to_astropy_table(cts)
                out_table.write(outfile, overwrite=True, **additional_output_args)
                filelist.append(outfile)

            if args.covariance:
                cov = CovarianceSpectrum(
                    events,
                    freq_interval=args.freq_interval,
                    energy_spec=energy_spec,
                    ref_band=args.ref_band,
                    segment_size=args.segment_size,
                    bin_time=args.bintime,
                    events2=events2,
                    use_pi=args.use_pi,
                    norm=args.norm,
                )
                outfile = hen_root(fname) + label + "_cov." + args.format
                out_table = varenergy_to_astropy_table(cov)
                out_table.write(outfile, overwrite=True, **additional_output_args)

                filelist.append(outfile)

    return filelist
