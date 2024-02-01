import os
import numpy as np
import pytest
from hendrics.calibrate import default_nustar_rmf
import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics import io, lcurve, read_events, fake
from hendrics.io import load_events, save_events
from . import cleanup_test_dir

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION


def test_default_nustar_rmf(caplog):
    caldb_path = "fake_caldb"
    os.environ["CALDB"] = caldb_path
    path_to_rmf = os.path.join(
        caldb_path, *"data/nustar/fpm/cpf/rmf/nuAdet3_20100101v002.rmf".split("/")
    )
    with pytest.warns(UserWarning, match="Using default NuSTAR rmf."):
        newpath = default_nustar_rmf()

    assert newpath == path_to_rmf


class TestCalibrate(object):
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
        cls.rmf = os.path.join(cls.datadir, "test.rmf")
        command = "{0} {1} ".format(
            os.path.join(cls.datadir, "monol_testA.evt"),
            os.path.join(cls.datadir, "monol_testB.evt"),
        )
        hen.read_events.main(command.split())
        command = "{} {} -r {} --nproc 2".format(
            os.path.join(
                cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
            ),
            os.path.join(
                cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION
            ),
            cls.rmf,
        )
        hen.calibrate.main(command.split())
        cls.xmm_fits_file = os.path.join(cls.datadir, "monol_test_fake_lc_xmm.evt")
        # Note that I don't specify the instrument. This is because
        # I want the internal machinery to understand that this is
        # XMM and this has to be given EPIC-pn by default.
        hen.fake.main(
            [
                "--deadtime",
                "1e-4",
                "-m",
                "XMM",
                "--ctrate",
                "2000",
                "--mjdref",
                "50814.0",
                "-o",
                cls.xmm_fits_file,
            ]
        )
        command = "{0}  --discard-calibration".format(cls.xmm_fits_file)
        hen.read_events.main(command.split())
        cls.xmm_ev_file = os.path.join(
            cls.datadir,
            "monol_test_fake_lc_xmm_xmm_epn_det01_ev" + HEN_FILE_EXTENSION,
        )

    def test_calibrate(self):
        """Test event file calibration."""
        from astropy.io.fits import Header

        ev = hen.io.load_events(self.ev_fileAcal)
        assert hasattr(ev, "header")

        Header.fromstring(ev.header)
        assert hasattr(ev, "gti")
        gti_to_test = hen.io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, ev.gti)

    def test_calibrate_xmm_raises(self):
        """Test event file calibration."""

        command = "{0} -r {1}".format(self.xmm_ev_file, self.rmf)

        with pytest.raises(RuntimeError, match="Calibration for XMM should work"):
            hen.calibrate.main(command.split())

    def test_calibrate_raises_missing_mission(self):
        """Test event file calibration."""
        from hendrics.io import load_events, save_events

        ev = load_events(self.ev_fileB)
        ev.mission = None
        bubu_fname = "budidum" + HEN_FILE_EXTENSION
        save_events(ev, bubu_fname)

        command = "{0} --rough".format(bubu_fname)

        with pytest.raises(ValueError):
            hen.calibrate.main(command.split())
        os.unlink(bubu_fname)

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
