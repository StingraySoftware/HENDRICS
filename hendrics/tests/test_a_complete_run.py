# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import shutil
import os
import glob
import subprocess as sp

import numpy as np
from astropy import log
from astropy.tests.helper import catch_warnings
from astropy.logger import AstropyUserWarning
from astropy.tests.helper import remote_data
import pytest
from stingray.lightcurve import Lightcurve
import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics.fold import HAS_PINT
from hendrics import (
    fake,
    fspec,
    base,
    binary,
    calibrate,
    colors,
    create_gti,
    exposure,
    exvar,
    io,
    lcurve,
    modeling,
    plot,
    read_events,
    rebin,
    save_as_xspec,
    timelags,
    varenergy,
    sum_fspec,
)

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION

log.setLevel("DEBUG")
# log.basicConfig(filename='HEN.log', level=log.DEBUG, filemode='w')


class TestFullRun(object):
    """Test how command lines work.

    Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/python-unittest-testcase-execution-order

    When command line is missing, uses some function calls
    """  # NOQA

    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        cls.ev_fileA = os.path.join(
            cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )
        cls.par = _dummy_par("bubububu.par")

        cls.ev_fileA = os.path.join(
            cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )
        cls.ev_fileB = os.path.join(
            cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION
        )
        cls.ev_fileAcal = os.path.join(
            cls.datadir,
            "monol_testA_nustar_fpma_ev_calib" + HEN_FILE_EXTENSION,
        )
        cls.ev_fileBcal = os.path.join(
            cls.datadir,
            "monol_testB_nustar_fpmb_ev_calib" + HEN_FILE_EXTENSION,
        )
        cls.par = _dummy_par("bubububu.par")
        command = "{0} {1}".format(
            os.path.join(cls.datadir, "monol_testA.evt"),
            os.path.join(cls.datadir, "monol_testB.evt"),
        )
        hen.read_events.main(command.split())
        command = "{} {} -r {}".format(
            os.path.join(
                cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
            ),
            os.path.join(
                cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION
            ),
            os.path.join(cls.datadir, "test.rmf"),
        )
        hen.calibrate.main(command.split())
        cls.lcA = os.path.join(
            os.path.join(
                cls.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION
            )
        )
        cls.lcB = os.path.join(
            os.path.join(
                cls.datadir, "monol_testB_E3-50_lc" + HEN_FILE_EXTENSION
            )
        )
        command = (
            "{} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " "-o {}"
        ).format(cls.ev_fileAcal, cls.lcA)
        hen.lcurve.main(command.split())
        command = (
            "{} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " "-o {}"
        ).format(cls.ev_fileBcal, cls.lcB)
        hen.lcurve.main(command.split())

        cls.pdsA = os.path.join(
            cls.datadir, "monol_testA_E3-50_pds" + HEN_FILE_EXTENSION
        )
        cls.pdsB = os.path.join(
            cls.datadir, "monol_testB_E3-50_pds" + HEN_FILE_EXTENSION
        )
        cls.cpds = os.path.join(
            cls.datadir, "monol_test_E3-50_cpds" + HEN_FILE_EXTENSION
        )

        command = "{} {} -f 128 -k PDS --save-all --norm leahy".format(
            cls.lcA, cls.lcB
        )
        hen.fspec.main(command.split())

        command = "{} {} -f 128 -k CPDS --save-all --norm leahy".format(
            cls.lcA, cls.lcB
        )
        hen.fspec.main(command.split())
        assert os.path.exists(cls.cpds)
        assert os.path.exists(cls.pdsA)
        assert os.path.exists(cls.pdsB)

    def test_scripts_are_installed(self):
        """Test only once that command line scripts are installed correctly."""
        fits_file = os.path.join(self.datadir, "monol_testA.evt")
        command = "HENreadfile {0}".format(fits_file)
        sp.check_call(command.split())

    def test_get_file_type(self):
        """Test getting file type."""
        file_list = {
            "events": "monol_testA_nustar_fpma_ev",
            "lc": "monol_testA_E3-50_lc",
            "pds": "monol_testA_E3-50_pds",
            "cpds": "monol_test_E3-50_cpds",
        }
        for realtype in file_list.keys():
            fname = os.path.join(
                self.datadir, file_list[realtype] + HEN_FILE_EXTENSION
            )
            ftype, _ = hen.io.get_file_type(fname)
            assert ftype == realtype, "File types do not match"

    def test_save_varen_rms(self):
        fname = self.ev_fileAcal
        hen.varenergy.main(
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
                "--rms",
                "-b",
                "0.5",
                "--segment-size",
                "128",
            ]
        )
        out = hen.base.hen_root(fname) + "_rms" + ".qdp"
        os.path.exists(out)

    def test_save_varen_lag(self):
        fname = self.ev_fileAcal
        hen.varenergy.main(
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
                "--lag",
                "-b",
                "0.5",
                "--segment-size",
                "128",
            ]
        )
        out = hen.base.hen_root(fname) + "_lag" + ".qdp"
        os.path.exists(out)

    def test_colors_fail_uncalibrated(self):
        """Test light curve using PI filtering."""
        command = ("{0} -b 100 -e {1} {2} {2} {3}").format(
            self.ev_fileA, 3, 5, 10
        )
        with pytest.raises(ValueError) as excinfo:
            hen.colors.main(command.split())

        assert "No energy information is present " in str(excinfo.value)

    def test_colors(self):
        """Test light curve using PI filtering."""
        # calculate colors
        command = ("{0} -b 100 -e {1} {2} {2} {3}").format(
            self.ev_fileAcal, 3, 5, 10
        )
        hen.colors.main(command.split())

        new_filename = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_E_10-5_over_5-3" + HEN_FILE_EXTENSION,
        )

        assert os.path.exists(new_filename)
        out_lc = hen.io.load_lcurve(new_filename)
        gti_to_test = hen.io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def test_readfile_fits(self):
        """Test reading and dumping a FITS file."""
        fitsname = os.path.join(self.datadir, "monol_testA.evt")
        command = "{0}".format(fitsname)

        hen.io.main(command.split())

    def test_plot_color(self):
        """Test plotting with linear axes."""
        lname = (
            os.path.join(
                self.datadir, "monol_testA_nustar_fpma_E_10-5_over_5-3"
            )
            + HEN_FILE_EXTENSION
        )
        cname = (
            os.path.join(
                self.datadir, "monol_testA_nustar_fpma_E_10-5_over_5-3"
            )
            + HEN_FILE_EXTENSION
        )
        hen.plot.main(
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
        command = ("{0} -b 100 --energy-interval {1} {2}").format(
            os.path.join(
                self.datadir,
                "monol_testA_nustar_fpma_ev_calib" + HEN_FILE_EXTENSION,
            ),
            3,
            10,
        )

        hen.lcurve.main(command.split())
        lname = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_E3-10_lc")
            + HEN_FILE_EXTENSION
        )
        os.path.exists(lname)
        cname = (
            os.path.join(
                self.datadir, "monol_testA_nustar_fpma_E_10-5_over_5-3"
            )
            + HEN_FILE_EXTENSION
        )
        hen.plot.main(
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

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        def find_file_pattern_in_dir(pattern, directory):
            return glob.glob(os.path.join(directory, pattern))

        patterns = [
            "*monol_test*" + HEN_FILE_EXTENSION,
            "*lcurve*" + HEN_FILE_EXTENSION,
            "*lcurve*.txt",
            "*.log",
            "*monol_test*.dat",
            "*monol_test*.png",
            "*monol_test*.txt",
            "*monol_test_fake*.evt",
            "*bubu*",
            "*.p",
            "*.qdp",
            "*.inf",
        ]

        file_list = []
        for pattern in patterns:
            file_list.extend(find_file_pattern_in_dir(pattern, self.datadir))

        for f in file_list:
            if os.path.exists(f):
                print("Removing " + f)
                os.remove(f)

        patterns = ["*_pds*/", "*_cpds*/", "*_sum/"]

        dir_list = []
        for pattern in patterns:
            dir_list.extend(find_file_pattern_in_dir(pattern, self.datadir))
        for f in dir_list:
            if os.path.exists(f):
                shutil.rmtree(f)
