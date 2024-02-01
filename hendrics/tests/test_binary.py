# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import os

import pytest

import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics.fold import HAS_PINT
from hendrics import binary, calibrate, io, lcurve, read_events
from . import cleanup_test_dir

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class TestBinary(object):
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
        cls.par = _dummy_par("bubububu.par")
        command = "{0} --discard-calibration".format(
            os.path.join(cls.datadir, "monol_testA.evt")
        )
        hen.read_events.main(command.split())
        command = "{} -r {}".format(
            os.path.join(
                cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
            ),
            os.path.join(cls.datadir, "test.rmf"),
        )
        hen.calibrate.main(command.split())
        cls.lcA = os.path.join(
            os.path.join(cls.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        )
        command = (
            "{} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " "-o {}"
        ).format(cls.ev_fileAcal, cls.lcA)
        hen.lcurve.main(command.split())

    def test_save_binary_events(self):
        f = self.ev_fileA
        with pytest.raises(ValueError, match="Energy filtering requested"):
            hen.binary.main_presto("{} -b 0.1 -e 3 59 --debug".format(f).split())

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_save_binary_calibrated_events(self):
        f = self.ev_fileAcal
        hen.binary.main_presto(
            "{} -b 0.1 -e 3 59 --debug --deorbit-par {}".format(f, self.par).split()
        )
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".dat"))
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".inf"))

    def test_save_binary_lc(self):
        f = self.lcA
        hen.binary.main_presto("{}".format(f).split())
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".dat"))
        assert os.path.exists(f.replace(HEN_FILE_EXTENSION, ".inf"))

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
