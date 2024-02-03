# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import shutil
import os
import glob
import subprocess as sp

import numpy as np
from astropy import log
from astropy.logger import AstropyUserWarning
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
from . import cleanup_test_dir

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
    Inspired by https://stackoverflow.com/questions/5387299/python-unittest-testcase-execution-order

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
        command = "{0} {1} --discard-calibration".format(
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

        cls.lc3_10 = os.path.join(
            os.path.join(cls.datadir, "monol_testA_E3-10_lc" + HEN_FILE_EXTENSION)
        )

        command = ("{} -e 3 10 -b 100 " "-o {}").format(cls.ev_fileAcal, cls.lc3_10)
        hen.lcurve.main(command.split())

        command = ("{0} -b 100 -e {1} {2} {2} {3}").format(cls.ev_fileAcal, 3, 5, 10)
        hen.colors.main(command.split())

        cls.colorfile = os.path.join(
            cls.datadir,
            "monol_testA_nustar_fpma_E_10-5_over_5-3" + HEN_FILE_EXTENSION,
        )

    def test_colors_correct_gti(self):
        """Test light curve using PI filtering."""
        # calculate colors

        assert os.path.exists(self.colorfile)
        out_lc = hen.io.load_lcurve(self.colorfile)
        gti_to_test = hen.io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def test_colors_fail_uncalibrated(self):
        """Test light curve using PI filtering."""
        command = ("{0} -b 100 -e {1} {2} {2} {3}").format(self.ev_fileA, 3, 5, 10)
        with pytest.raises(ValueError, match="Energy information not found in file"):
            hen.colors.main(command.split())

    def test_plot_color(self):
        """Test plotting with linear axes."""
        lname = self.colorfile
        cname = self.colorfile
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
            self.ev_fileAcal, 3, 10
        )
        hen.lcurve.main(command.split())

        lname = self.lc3_10
        cname = self.colorfile
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
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
