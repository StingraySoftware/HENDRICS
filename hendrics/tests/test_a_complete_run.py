# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import os
import subprocess as sp

import numpy as np
import pytest

from astropy import log
from astropy.io.registry import IORegistryError
from hendrics import (
    base,
    calibrate,
    colors,
    fspec,
    io,
    lcurve,
    plot,
    power_colors,
    read_events,
    varenergy,
)
from hendrics.io import HAS_H5PY
from hendrics.tests import _dummy_par

from . import cleanup_test_dir, find_file_pattern_in_dir

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

HEN_FILE_EXTENSION = io.HEN_FILE_EXTENSION

log.setLevel("DEBUG")
# log.basicConfig(filename='HEN.log', level=log.DEBUG, filemode='w')


class TestFullRun:
    """Test how command lines work.

    Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by https://stackoverflow.com/questions/5387299/python-unittest-testcase-execution-order

    When command line is missing, uses some function calls
    """

    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        cls.ev_fileA = os.path.join(cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION)
        cls.par = _dummy_par("bubububu.par")

        cls.ev_fileA = os.path.join(cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION)
        cls.ev_fileB = os.path.join(cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION)
        cls.ev_fileAcal = os.path.join(
            cls.datadir,
            "monol_testA_nustar_fpma_ev_calib" + HEN_FILE_EXTENSION,
        )
        cls.ev_fileBcal = os.path.join(
            cls.datadir,
            "monol_testB_nustar_fpmb_ev_calib" + HEN_FILE_EXTENSION,
        )
        cls.par = _dummy_par("bubububu.par")
        data_a, data_b = (
            os.path.join(cls.datadir, "monol_testA.evt"),
            os.path.join(cls.datadir, "monol_testB.evt"),
        )
        command = f"{data_a} {data_b} --discard-calibration"
        read_events.main(command.split())

        data_a, data_b, rmf = (
            os.path.join(cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir, "test.rmf"),
        )
        command = f"{data_a} {data_b} -r {rmf}"
        calibrate.main(command.split())
        cls.lcA = os.path.join(
            os.path.join(cls.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        )
        cls.lcB = os.path.join(
            os.path.join(cls.datadir, "monol_testB_E3-50_lc" + HEN_FILE_EXTENSION)
        )
        command = (
            f"{cls.ev_fileAcal} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " f"-o {cls.lcA}"
        )
        lcurve.main(command.split())
        command = (
            f"{cls.ev_fileBcal} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " f"-o {cls.lcB}"
        )
        lcurve.main(command.split())

        cls.pdsA = os.path.join(
            cls.datadir, "monol_testA_E3-50_0d000244141_128_leahy_pds" + HEN_FILE_EXTENSION
        )
        cls.pdsB = os.path.join(
            cls.datadir, "monol_testB_E3-50_0d000244141_128_leahy_pds" + HEN_FILE_EXTENSION
        )
        cls.cpds = os.path.join(
            cls.datadir, "monol_test_E3-50_0d000244141_128_leahy_cpds" + HEN_FILE_EXTENSION
        )

        command = f"{cls.lcA} {cls.lcB} -f 128 -k PDS --save-all --norm leahy"
        fspec.main(command.split())

        command = f"{cls.lcA} {cls.lcB} -f 128 -k CPDS --save-all --norm leahy"
        fspec.main(command.split())
        import glob

        print(glob.glob(os.path.join(cls.datadir, "monol_test*_E3-50_*pds*")))
        assert os.path.exists(cls.cpds)
        assert os.path.exists(cls.pdsA)
        assert os.path.exists(cls.pdsB)

    def test_scripts_are_installed(self):
        """Test only once that command line scripts are installed correctly."""
        fits_file = os.path.join(self.datadir, "monol_testA.evt")
        command = f"HENreadfile {fits_file}"
        sp.check_call(command.split())

    def test_get_file_type(self):
        """Test getting file type."""
        file_list = {
            "events": "monol_testA_nustar_fpma_ev",
            "lc": "monol_testA_E3-50_lc",
            "pds": "monol_testA_E3-50_0d000244141_128_leahy_pds",
            "cpds": "monol_test_E3-50_0d000244141_128_leahy_cpds",
        }
        for realtype in file_list.keys():
            fname = os.path.join(self.datadir, file_list[realtype] + HEN_FILE_EXTENSION)
            ftype, _ = io.get_file_type(fname)
            assert ftype == realtype, "File types do not match"

    @pytest.mark.parametrize("format", ["qdp", "ecsv", "csv", "hdf5"])
    @pytest.mark.parametrize("kind", ["rms", "cov", "count", "lag"])
    def test_save_varen(self, kind, format):
        fname = self.ev_fileAcal
        if not HAS_H5PY and format == "hdf5":
            return

        try:
            varenergy.main(
                [
                    fname,
                    "-f",
                    "0",
                    "100",
                    "--energy-values",
                    "0.3",
                    "12",
                    "5",
                    "lin",
                    f"--{kind}",
                    "-b",
                    "0.5",
                    "--segment-size",
                    "128",
                    "--format",
                    format,
                    "--label",
                    "nice",
                ]
            )
            out = base.hen_root(fname) + f"_nice_{kind}" + f".{format}"
            assert os.path.exists(out)
        except IORegistryError:
            pass

    def test_colors_fail_uncalibrated(self):
        """Test light curve using PI filtering."""
        command = f"{self.ev_fileA} -b 100 -e 3 5 5 10"
        with pytest.raises(ValueError, match="Energy information not found in file"):
            colors.main(command.split())

    def test_colors(self):
        """Test light curve using PI filtering."""
        # calculate colors
        command = f"{self.ev_fileAcal} -b 100 -e 3 5 5 10"
        colors.main(command.split())

        new_filename = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_E_10-5_over_5-3" + HEN_FILE_EXTENSION,
        )

        assert os.path.exists(new_filename)
        out_lc = io.load_lcurve(new_filename)
        gti_to_test = io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def test_power_colors(self):
        """Test light curve using PI filtering."""
        # calculate colors
        command = f"{self.ev_fileAcal} --debug -s 16 -b -6 -f 1 2 4 8 16 "
        with pytest.warns(
            UserWarning,
            match="(Some .non-log. power colors)|(All power spectral)|(Poisson-subtracted power)",
        ):
            new_filenames = power_colors.main(command.split())

        assert os.path.exists(new_filenames[0])
        plot.main(new_filenames)

    def test_power_colors_2files(self):
        """Test light curve using PI filtering."""
        # calculate colors
        command = f"--cross {self.ev_fileAcal} {self.ev_fileBcal} -s 16 -b -6 -f 1 2 4 8 16 "
        with pytest.warns(
            UserWarning,
            match="(Some .non-log.)|(All power spectral)|(Poisson-subtracted)|(cast to real)",
        ):
            new_filenames = power_colors.main(command.split())

        assert os.path.exists(new_filenames[0])
        plot.main(new_filenames)

    def test_power_colors_2files_raises_no_cross_output(self):
        """Test light curve using PI filtering."""
        # calculate colors
        with pytest.raises(ValueError, match="Specify --output only when processing"):
            command = f"{self.ev_fileAcal} {self.ev_fileBcal} -s 16 -b -6 -f 1 2 4 8 16 -o bubu.nc"
            power_colors.main(command.split())

    def test_readfile_fits(self):
        """Test reading and dumping a FITS file."""
        fitsname = os.path.join(self.datadir, "monol_testA.evt")
        command = f"{fitsname}"

        io.main(command.split())

    def test_plot_color(self):
        """Test plotting with linear axes."""
        lname = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_E_10-5_over_5-3")
            + HEN_FILE_EXTENSION
        )
        cname = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_E_10-5_over_5-3")
            + HEN_FILE_EXTENSION
        )
        plot.main(
            [
                cname,
                lname,
                "--noplot",
                "--xlog",
                "--ylog",
                "--CCD",
                "-o",
                "dummy.qdp",
            ]
        )

    def test_plot_hid(self):
        """Test plotting with linear axes."""
        # also produce a light curve with the same binning
        data = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_ev_calib" + HEN_FILE_EXTENSION,
        )
        command = f"{data} -b 100 --energy-interval 3 10"

        lcurve.main(command.split())
        import glob

        print(
            glob.glob(
                os.path.join(
                    self.datadir,
                    "*" + HEN_FILE_EXTENSION,
                )
            )
        )
        lname = os.path.join(self.datadir, "monol_testA_nustar_fpma_E3-10_lc") + HEN_FILE_EXTENSION
        assert os.path.exists(lname)
        cname = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_E_10-5_over_5-3")
            + HEN_FILE_EXTENSION
        )
        plot.main(
            [
                cname,
                lname,
                "--noplot",
                "--xlog",
                "--ylog",
                "--HID",
                "-o",
                "dummy.qdp",
            ]
        )

    def test_all_files_get_read(self):
        # Test that HENreadfile works with all file types
        fnames = find_file_pattern_in_dir("*" + HEN_FILE_EXTENSION, self.datadir)
        io.main(fnames)

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        cleanup_test_dir(self.datadir)
        cleanup_test_dir(".")
