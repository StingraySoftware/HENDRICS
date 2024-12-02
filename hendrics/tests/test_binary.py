# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import os

import pytest

from hendrics import binary, calibrate, io, lcurve, read_events
from hendrics.base import HAS_PINT
from hendrics.tests import _dummy_par

from . import cleanup_test_dir

HEN_FILE_EXTENSION = io.HEN_FILE_EXTENSION

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class TestBinary:
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
        cls.par = _dummy_par("bubububu.par")
        data = os.path.join(cls.datadir, "monol_testA.evt")
        command = f"{data} --discard-calibration"
        read_events.main(command.split())

        data, rmf = (
            os.path.join(cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir, "test.rmf"),
        )
        command = f"{data} -r {rmf}"
        calibrate.main(command.split())

        cls.lcA = os.path.join(
            os.path.join(cls.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        )
        command = (
            f"{cls.ev_fileAcal} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " f"-o {cls.lcA}"
        )
        lcurve.main(command.split())

    def test_save_binary_events(self):
        f = self.ev_fileA
        with pytest.raises(ValueError, match="Energy filtering requested"):
            binary.main_presto(f"{f} -b 0.1 -e 3 59 --debug".split())

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_save_binary_calibrated_events(self):
        f = self.ev_fileAcal
        binary.main_presto(f"{f} -b 0.1 -e 3 59 --debug --deorbit-par {self.par}".split())
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".dat"))
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".inf"))

    def test_save_binary_lc(self):
        f = self.lcA
        binary.main_presto(f"{f}".split())
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".dat"))
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".inf"))

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
