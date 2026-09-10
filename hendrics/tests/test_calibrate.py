import os

import numpy as np
import pytest

from hendrics import calibrate, fake, io, read_events
from hendrics.calibrate import default_nustar_rmf, read_calibration
from hendrics.io import load_events, save_events
from hendrics.tests import _dummy_par

from . import cleanup_test_dir

HEN_FILE_EXTENSION = io.HEN_FILE_EXTENSION


def test_default_nustar_rmf(caplog):
    caldb_path = "fake_caldb"
    os.environ["CALDB"] = caldb_path
    path_to_rmf = os.path.join(
        caldb_path, "data", "nustar", "fpm", "cpf", "rmf", "nuAdet3_20100101v002.rmf"
    )
    with pytest.warns(UserWarning, match="Using default NuSTAR rmf."):
        newpath = default_nustar_rmf()

    assert newpath == path_to_rmf


def test_default_nustar_rmf_without_caldb_raises(monkeypatch):
    """Without CALDB there is nowhere to look, so say so instead of KeyError."""
    monkeypatch.delenv("CALDB", raising=False)
    with pytest.warns(UserWarning, match="Using default NuSTAR rmf."):
        with pytest.raises(ValueError, match="CALDB environment variable is not set"):
            default_nustar_rmf()


def test_read_calibration_channels_missing_from_rmf(monkeypatch):
    """Channels absent from the RMF get no energy, not a neighbour's energy.

    ``read_calibration`` looks the channels up with ``searchsorted``, which
    returns the position a value *would* take; unmatched channels have to be
    dropped again afterwards.
    """
    curdir = os.path.abspath(os.path.dirname(__file__))
    rmf = os.path.join(curdir, "data", "test.rmf")

    # test.rmf covers channels 0-4095
    pis = np.array([0, 10, 4095, 4096, 100000, -3])
    es = read_calibration(pis, rmf)

    assert np.all(es[:3] > 0)
    assert np.all(es[3:] == 0)

    # An RMF with no EBOUNDS at all: everything stays at zero
    monkeypatch.setattr(
        "hendrics.calibrate.read_rmf",
        lambda _: (np.array([]), np.array([]), np.array([])),
    )
    es = read_calibration(pis, rmf)
    assert es.shape == pis.shape
    assert np.all(es == 0)


class TestCalibrate:
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
        cls.rmf = os.path.join(cls.datadir, "test.rmf")
        data_a, data_b = (
            os.path.join(cls.datadir, "monol_testA.evt"),
            os.path.join(cls.datadir, "monol_testB.evt"),
        )
        command = f"{data_a} {data_b} ".format()
        read_events.main(command.split())

        data_a, data_b, rmf = (
            os.path.join(cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION),
            cls.rmf,
        )
        command = f"{data_a} {data_b} -r {rmf} --nproc 2"
        calibrate.main(command.split())
        cls.xmm_fits_file = os.path.join(cls.datadir, "monol_test_fake_lc_xmm.evt")
        # Note that I don't specify the instrument. This is because
        # I want the internal machinery to understand that this is
        # XMM and this has to be given EPIC-pn by default.
        fake.main(
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
        command = f"{cls.xmm_fits_file}  --discard-calibration"
        read_events.main(command.split())
        cls.xmm_ev_file = os.path.join(
            cls.datadir,
            "monol_test_fake_lc_xmm_xmm_epn_det01_ev" + HEN_FILE_EXTENSION,
        )

    def test_calibrate(self):
        """Test event file calibration."""
        from astropy.io.fits import Header

        ev = io.load_events(self.ev_fileAcal)
        assert hasattr(ev, "header")

        Header.fromstring(ev.header)
        assert hasattr(ev, "gti")
        gti_to_test = io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, ev.gti)

    def test_calibrate_xmm_raises(self):
        """Test event file calibration."""

        command = f"{self.xmm_ev_file} -r {self.rmf}"

        with pytest.raises(RuntimeError, match="Calibration for XMM should work"):
            calibrate.main(command.split())

    def test_calibrate_raises_missing_mission(self):
        """Test event file calibration."""

        ev = load_events(self.ev_fileB)
        ev.mission = None
        bubu_fname = "budidum" + HEN_FILE_EXTENSION
        save_events(ev, bubu_fname)

        command = f"{bubu_fname} --rough"

        with pytest.raises(ValueError):
            calibrate.main(command.split())
        os.unlink(bubu_fname)

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
