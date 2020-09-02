import os
import numpy as np
import pytest
from hendrics.calibrate import default_nustar_rmf
import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics import io, lcurve, read_events

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION


def test_default_nustar_rmf(caplog):
    caldb_path = "fake_caldb"
    os.environ["CALDB"] = caldb_path
    path_to_rmf = os.path.join(
        caldb_path,
        *"data/nustar/fpm/cpf/rmf/nuAdet3_20100101v002.rmf".split("/")
    )
    with pytest.warns(UserWarning) as record:
        newpath = default_nustar_rmf()

    assert np.any(
        ["Using default NuSTAR rmf." in r.message.args[0] for r in record]
    )
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
            os.path.join(cls.datadir, "test.rmf"),
        )
        hen.calibrate.main(command.split())

    def test_calibrate(self):
        """Test event file calibration."""
        from astropy.io.fits import Header

        ev = hen.io.load_events(self.ev_fileAcal)
        assert hasattr(ev, "header")

        Header.fromstring(ev.header)
        assert hasattr(ev, "gti")
        gti_to_test = hen.io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, ev.gti)

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
