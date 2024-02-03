# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import shutil
import os
import glob
from astropy import log

import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics import calibrate, create_gti, io, lcurve, read_events
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
        command = "{0} {1}".format(
            os.path.join(cls.datadir, "monol_testA.evt"),
            os.path.join(cls.datadir, "monol_testB.evt"),
        )
        hen.read_events.main(command.split())
        command = "{} {} -r {}".format(
            cls.ev_fileA, cls.ev_fileB, os.path.join(cls.datadir, "test.rmf")
        )
        hen.calibrate.main(command.split())
        cls.lcA = os.path.join(
            os.path.join(cls.datadir, "monol_testA_lc" + HEN_FILE_EXTENSION)
        )
        cls.lcB = os.path.join(
            os.path.join(cls.datadir, "monol_testB_lc" + HEN_FILE_EXTENSION)
        )
        command = ("{}  --nproc 2 -b 2 " "-o {}").format(cls.ev_fileAcal, cls.lcA)
        hen.lcurve.main(command.split())
        command = ("{}  --nproc 2 -b 2 " "-o {}").format(cls.ev_fileBcal, cls.lcB)
        hen.lcurve.main(command.split())

        command = "{0} -f time>0 -c --debug".format(cls.ev_fileA)
        hen.create_gti.main(command.split())
        cls.gtifile = (
            os.path.join(cls.datadir, "monol_testA_nustar_fpma_gti")
            + HEN_FILE_EXTENSION
        )

    def test_create_gti(self):
        """Test creating a GTI file."""
        assert os.path.exists(self.gtifile)

    def test_apply_gti(self):
        """Test applying a GTI file."""
        fname = self.gtifile
        lcfname = self.ev_fileA
        lcoutname = self.ev_fileA.replace(
            HEN_FILE_EXTENSION, "_gtifilt" + HEN_FILE_EXTENSION
        )
        command = "{0} -a {1} --debug".format(lcfname, fname)
        hen.create_gti.main(command.split())
        hen.io.load_events(lcoutname)

    def test_create_gti_and_minlen(self):
        """Test creating a GTI file and apply minimum length."""
        fname = self.lcA
        command = "{0} -f counts>0 -c -l 10 --debug".format(fname)
        hen.create_gti.main(command.split())

    def test_create_gti_and_apply(self):
        """Test applying a GTI file and apply minimum length."""
        fname = self.gtifile
        lcfname = self.lcA
        command = "{0} -a {1} -l 10 --debug".format(lcfname, fname)
        hen.create_gti.main(command.split())

    def test_readfile(self):
        """Test reading and dumping a HENDRICS file."""
        fname = self.gtifile
        command = "{0}".format(fname)

        hen.io.main(command.split())

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
