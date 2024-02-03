# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import shutil
import os
import glob
import subprocess as sp
import pytest
import numpy as np
from astropy import log
from astropy.io import fits
import hendrics as hen
from stingray.events import EventList
from hendrics.tests import _dummy_par
from hendrics import fake, calibrate, read_events, io
from hendrics.fake import scramble
from hendrics.io import load_events
from hendrics.fold import HAS_PINT
from . import cleanup_test_dir

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION

log.setLevel("DEBUG")
# log.basicConfig(filename='HEN.log', level=log.DEBUG, filemode='w')


def test_filter_for_deadtime_nonpar():
    """Test dead time filter, non-paralyzable case."""
    events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
    filt_events = hen.fake.filter_for_deadtime(events, 0.11)
    expected = np.array([1, 2, 2.2, 3, 3.2])
    assert np.all(filt_events == expected), "Wrong: {} vs {}".format(
        filt_events, expected
    )


def test_filter_for_deadtime_nonpar_bkg():
    """Test dead time filter, non-paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = hen.fake.filter_for_deadtime(
        events, 0.11, bkg_ev_list=bkg_events, return_all=True
    )
    expected_ev = np.array([2, 2.2, 3, 3.2])
    expected_bk = np.array([1])
    assert np.all(filt_events == expected_ev), "Wrong: {} vs {}".format(
        filt_events, expected_ev
    )
    assert np.all(info.bkg == expected_bk), "Wrong: {} vs {}".format(
        info.bkg, expected_bk
    )


def test_filter_for_deadtime_par():
    """Test dead time filter, paralyzable case."""
    events = np.array([1, 1.1, 2, 2.2, 3, 3.1, 3.2])
    assert np.all(
        hen.fake.filter_for_deadtime(events, 0.11, paralyzable=True)
        == np.array([1, 2, 2.2, 3])
    )


def test_filter_for_deadtime_par_bkg():
    """Test dead time filter, paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = hen.fake.filter_for_deadtime(
        events,
        0.11,
        bkg_ev_list=bkg_events,
        paralyzable=True,
        return_all=True,
    )
    expected_ev = np.array([2, 2.2, 3])
    expected_bk = np.array([1])
    assert np.all(filt_events == expected_ev), "Wrong: {} vs {}".format(
        filt_events, expected_ev
    )
    assert np.all(info.bkg == expected_bk), "Wrong: {} vs {}".format(
        info.bkg, expected_bk
    )


def test_filter_for_deadtime_par_bkg_obj():
    """Test dead time filter on Eventlist, paral. case, with background."""
    times = np.array([1.1, 2, 2.2, 3, 3.2])
    pis = np.arange(times.size) + 5
    energies = np.arange(times.size) + 10
    events = EventList(time=times, energy=energies, pi=pis)
    bkg_events = np.array([1, 3.1])
    filt_events, info = hen.fake.filter_for_deadtime(
        events,
        0.11,
        bkg_ev_list=bkg_events,
        paralyzable=True,
        return_all=True,
    )
    expected_ev = np.array([2, 2.2, 3])
    expected_bk = np.array([1])
    expected_pi = np.array([6, 7, 8])
    expected_nrg = np.array([11, 12, 13])
    filt_times = filt_events.time
    filt_pis = filt_events.pi
    filt_nrgs = filt_events.energy

    assert np.all(filt_times == expected_ev), "Wrong: {} vs {}".format(
        filt_events, expected_ev
    )
    assert np.all(filt_pis == expected_pi), "Wrong: {} vs {}".format(
        filt_events, expected_ev
    )
    assert np.all(filt_nrgs == expected_nrg), "Wrong: {} vs {}".format(
        filt_events, expected_ev
    )
    assert np.all(info.bkg == expected_bk), "Wrong: {} vs {}".format(
        info.bkg, expected_bk
    )


def test_deadtime_mask_par():
    """Test dead time filter, paralyzable case, with background."""
    events = np.array([1.1, 2, 2.2, 3, 3.2])
    bkg_events = np.array([1, 3.1])
    filt_events, info = hen.fake.filter_for_deadtime(
        events,
        0.11,
        bkg_ev_list=bkg_events,
        paralyzable=True,
        return_all=True,
    )

    assert np.allclose(filt_events, [2.0, 2.2, 3.0])


def test_deadtime_conversion():
    """Test the functions for count rate conversion."""
    original_rate = np.arange(1, 1000, 10)
    deadtime = 2.5e-3
    rdet = hen.base.r_det(deadtime, original_rate)
    rin = hen.base.r_in(deadtime, rdet)
    np.testing.assert_almost_equal(rin, original_rate)


def verify_all_checksums(filename):
    with fits.open(filename) as hdul:
        for hdu in hdul:
            assert hdu.verify_datasum() == 1, f"Bad datasum: {hdu.name}"
            assert hdu.verify_checksum() == 1, f"Bad checksum: {hdu.name}"


class TestFake(object):
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
        cls.first_event_file = os.path.join(
            cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )
        cls.par = _dummy_par("bubububu.par")
        cls.fits_fileA = os.path.join(cls.datadir, "monol_testA.evt")
        command = "{0} --discard-calibration".format(cls.fits_fileA)
        hen.read_events.main(command.split())

        cls.first_event_file_cal = "calibrated" + HEN_FILE_EXTENSION
        hen.calibrate.calibrate(
            cls.first_event_file, cls.first_event_file_cal, rough=True
        )

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

    def test_checksums(self):
        for fname in [self.xmm_fits_file, self.fits_fileA]:
            verify_all_checksums(fname)

    def test_fake_file(self):
        """Test produce a fake event file."""
        fits_file = os.path.join(self.datadir, "monol_test_fake.evt")
        hen.fake.main(["-o", fits_file, "--instrument", "FPMB"])
        verify_all_checksums(fits_file)
        info = hen.io.print_fits_info(fits_file, hdu=1)
        assert info["Instrument"] == "FPMB"

    def test_fake_file_from_input_lc(self):
        """Test produce a fake event file from input light curve."""
        lcurve_in = os.path.join(self.datadir, "lcurveA.fits")
        fits_file = os.path.join(self.datadir, "monol_test_fake_lc.evt")
        with pytest.warns(UserWarning, match="FITS light curve handling is st"):
            hen.fake.main(["--lc", lcurve_in, "-o", fits_file])

        verify_all_checksums(fits_file)

    def test_fake_file_with_deadtime(self):
        """Test produce a fake event file and apply deadtime."""
        fits_file = os.path.join(self.datadir, "monol_test_fake_lc.evt")
        hen.fake.main(["--deadtime", "2.5e-3", "--ctrate", "2000", "-o", fits_file])
        verify_all_checksums(fits_file)

    def test_fake_file_xmm(self):
        """Test produce a fake event file and apply deadtime."""

        with fits.open(self.xmm_fits_file) as hdu_list:
            hdunames = [hdu.name for hdu in hdu_list]
            assert "STDGTI01" in hdunames
            assert "STDGTI02" in hdunames
            assert "STDGTI07" in hdunames
            assert hdu_list[0].header["TELESCOP"].lower() == "xmm"
            assert hdu_list[0].header["INSTRUME"].lower() == "epn"

        assert os.path.exists(self.xmm_ev_file)

    def test_load_events_randomize(self):
        """Test event file reading."""
        newfiles = hen.read_events.treat_event_file(self.fits_fileA, randomize_by=0.073)
        clean_file = self.first_event_file
        ev_clean = hen.io.load_events(clean_file)
        ev = hen.io.load_events(newfiles[0])
        diff = ev.time - ev_clean.time
        assert np.all(np.abs(diff) <= 0.073 / 2)
        assert np.all(np.abs(diff) > 0.0)

    def test_scramble_events_file(self):
        command = f"{self.first_event_file}"
        newfile = hen.fake.main_scramble(command.split())
        assert os.path.exists(newfile)
        os.remove(newfile)

    @pytest.mark.parametrize("fname", ["first_event_file", "xmm_ev_file"])
    def test_fake_fits_input_events_file(self, fname):
        newfile = "bububuasdf.fits"
        infname = getattr(self, fname)
        command = f"-e {infname} -o {newfile}"
        _ = hen.fake.main(command.split())
        assert os.path.exists(newfile)

        verify_all_checksums(newfile)

        events0 = load_events(infname)
        newf = hen.read_events.treat_event_file(newfile)

        events1 = load_events(newf[0])

        assert np.allclose(events0.gti, events1.gti)
        assert np.allclose(events0.time, np.sort(events0.time))
        assert np.allclose(events1.time, np.sort(events1.time))
        assert np.allclose(events0.time, events1.time)

        assert np.isclose(events0.mjdref, events1.mjdref)
        if hasattr(events0, "detector_id") and events0.detector_id is not None:
            assert np.allclose(events0.detector_id, events1.detector_id)

        if "xmm" in fname:
            assert np.isclose(events0.mjdref, 50814)
        os.remove(newfile)

    def test_scramble_uncalibrated_events_file_raises(self):
        command = f"{self.first_event_file} -e 3 30"
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning, match="No energy information"):
                _ = hen.fake.main_scramble(command.split())

    def test_scramble_calibrated_events_file(self):
        command = f"{self.first_event_file_cal} -e 3 30"
        newfile = hen.fake.main_scramble(command.split())
        assert "3-30" in newfile
        assert os.path.exists(newfile)
        os.remove(newfile)

    @pytest.mark.remote_data
    @pytest.mark.skipif("not HAS_PINT")
    def test_scramble_events_file_deorbit(self):
        _ = _dummy_par("bububububu.par", pb=1.0, a1=30)
        command = f"{self.first_event_file} --deorbit-par bububububu.par"
        newfile = hen.fake.main_scramble(command.split())
        assert os.path.exists(newfile)
        os.remove(newfile)

    def test_scramble_events(self):
        nevents = 3003
        times = np.random.uniform(0, 1000, nevents)
        times = times[(times > 125.123)]
        # Put exactly one photon inside a very short GTI
        times[0] = 0.5
        times = np.sort(times)
        event_list = EventList(
            times, gti=np.array([[0, 0.9], [111, 123.2], [125.123, 1000]])
        )

        new_event_list = scramble(event_list, "smooth")
        assert new_event_list.time.size == times.size
        assert np.all(new_event_list.gti == event_list.gti)

        new_event_list = scramble(event_list, "flat")
        assert new_event_list.time.size == times.size
        np.all(new_event_list.gti == event_list.gti)

    def test_calibrate_xmm(self):
        """Test event file calibration."""
        xmm_file = self.xmm_ev_file
        command = "{0} -r {1} --nproc 2".format(
            xmm_file, os.path.join(self.datadir, "test.rmf")
        )
        with pytest.raises(RuntimeError):
            hen.calibrate.main(command.split())

    def test_calibrate_xmm_normf(self):
        """Test event file calibration."""
        xmm_file = self.xmm_ev_file
        command = "{0} --rough --nproc 2".format(xmm_file)
        hen.calibrate.main(command.split())

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
