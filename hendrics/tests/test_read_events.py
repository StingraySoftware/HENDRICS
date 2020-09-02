# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import glob
import pytest
import numpy as np
from astropy.tests.helper import catch_warnings

from stingray.events import EventList
from hendrics.read_events import treat_event_file
from hendrics.io import HEN_FILE_EXTENSION, load_data, save_events, load_events
from hendrics.io import ref_mjd
from hendrics.fake import main
import hendrics as hen


class TestMergeEvents:
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")

        ev0 = EventList(
            time=np.sort(np.random.uniform(0, 100, 10)),
            gti=np.array([[0.0, 100]]),
            mjdref=1,
        )
        ev1 = EventList(
            time=np.sort(np.random.uniform(200, 300, 10)),
            gti=np.array([[200.0, 300]]),
            mjdref=2,
        )
        ev2 = EventList(
            time=np.sort(np.random.uniform(400, 500, 10)),
            gti=np.array([[400.0, 500]]),
            mjdref=1,
        )
        ev3 = EventList(
            time=np.sort(np.random.uniform(600, 700, 10)),
            gti=np.array([[600.0, 700]]),
            mjdref=1,
        )
        ev4 = EventList(
            time=np.sort(np.random.uniform(0, 100, 10)),
            gti=np.array([[600.0, 700]]),
            mjdref=1,
        )

        for ev in [ev0, ev1, ev2, ev3]:
            ev.pi = np.random.randint(0, 10, ev.time.size)
            ev.energy = np.random.uniform(3, 79, ev.time.size)

        ev0.instr = ev1.instr = ev2.instr = ev4.instr = "BA"
        ev3.instr = "BU"
        f0 = os.path.join(cls.datadir, "ev0_ev" + HEN_FILE_EXTENSION)
        f1 = os.path.join(cls.datadir, "ev1_ev" + HEN_FILE_EXTENSION)
        f2 = os.path.join(cls.datadir, "ev2_ev" + HEN_FILE_EXTENSION)
        f3 = os.path.join(cls.datadir, "ev3_ev" + HEN_FILE_EXTENSION)
        f4 = os.path.join(cls.datadir, "ev4_ev" + HEN_FILE_EXTENSION)

        save_events(ev0, f0)
        save_events(ev1, f1)
        save_events(ev2, f2)
        save_events(ev3, f3)
        save_events(ev4, f4)

        cls.f0, cls.f1, cls.f2, cls.f3, cls.f4 = f0, f1, f2, f3, f4

    def test_merge_events(self):
        hen.read_events.main_join(
            [
                self.f0,
                self.f1,
                "-o",
                os.path.join(
                    self.datadir, "monol_merg_ev" + HEN_FILE_EXTENSION
                ),
            ]
        )

        out = os.path.join(self.datadir, "monol_merg_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)
        os.unlink(out)

    def test_merge_events_no_out_fname(self):
        hen.read_events.main_join([self.f0, self.f1])

        out = os.path.join(self.datadir, "ev_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)
        os.unlink(out)

    def test_merge_many_events_warnings(self):

        out = os.path.join(
            self.datadir, "monol_merg_many_ev" + HEN_FILE_EXTENSION
        )
        with pytest.warns(UserWarning) as record:
            hen.read_events.main_join([self.f0, self.f1, self.f2, "-o", out])
        assert np.any(
            [
                f"{self.f1} has a different MJDREF" in r.message.args[0]
                for r in record
            ]
        )
        assert os.path.exists(out)
        os.unlink(out)
        with pytest.warns(UserWarning) as record:
            hen.read_events.main_join([self.f0, self.f2, self.f3, "-o", out])
        assert np.any(
            [
                f"{self.f3} is from a different" in r.message.args[0]
                for r in record
            ]
        )
        assert os.path.exists(out)
        os.unlink(out)
        with pytest.warns(UserWarning) as record:
            hen.read_events.main_join([self.f0, self.f2, self.f4, "-o", out])
        assert np.any(
            [
                f"{self.f4} has no good events" in r.message.args[0]
                for r in record
            ]
        )
        assert os.path.exists(out)
        os.unlink(out)

    def test_merge_many_events(self):
        outfile = "joint_ev" + HEN_FILE_EXTENSION
        # Note that only 0 and 2 are valid
        hen.read_events.main_join([self.f0, self.f2, self.f3])
        assert os.path.exists(outfile)

        data = load_events(outfile)
        assert hasattr(data, "gti")
        assert data.gti is not None
        allgtis = []
        # Note that only 0 and 2 are valid
        for evfile in [self.f0, self.f2]:
            ev = load_events(evfile)
            allgtis.append(ev.gti)
        allgtis = np.sort(np.concatenate(allgtis))
        assert np.allclose(data.gti, allgtis)
        os.unlink(outfile)


class TestReadEvents:
    """Real unit tests."""

    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        cls.fits_fileA = os.path.join(cls.datadir, "monol_testA.evt")
        cls.fits_fileB = os.path.join(cls.datadir, "monol_testB.evt")
        cls.fits_file = os.path.join(cls.datadir, "monol_test_fake.evt")
        main(
            [
                "--deadtime",
                "1e-4",
                "-m",
                "XMM",
                "-i",
                "epn",
                "--ctrate",
                "2000",
                "--mjdref",
                "50814.0",
                "-o",
                cls.fits_file,
            ]
        )
        cls.ev_fileA = os.path.join(
            cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )
        cls.ev_fileB = os.path.join(
            cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION
        )

    def test_treat_event_file_nustar(self):
        treat_event_file(self.fits_fileA)
        new_filename = "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir, new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert "instr" in data
        assert "gti" in data
        assert "mjdref" in data
        assert np.isclose(data["mjdref"], ref_mjd(self.fits_fileA))

    def test_treat_event_file_xmm(self):
        treat_event_file(self.fits_file)
        new_filename = "monol_test_fake_xmm_epn_det01_ev" + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir, new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert "instr" in data
        assert "gti" in data
        assert "mjdref" in data

    def test_treat_event_file_xmm_gtisplit(self):

        treat_event_file(self.fits_file, gti_split=True)
        new_filename = (
            "monol_test_fake_xmm_epn_det01_gti000_ev" + HEN_FILE_EXTENSION
        )
        assert os.path.exists(os.path.join(self.datadir, new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert "instr" in data
        assert "gti" in data
        assert "mjdref" in data

    def test_treat_event_file_xmm_lensplit(self):

        treat_event_file(self.fits_file, length_split=10)
        new_filename = (
            "monol_test_fake_xmm_epn_det01_chunk000_ev" + HEN_FILE_EXTENSION
        )
        assert os.path.exists(os.path.join(self.datadir, new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert "instr" in data
        assert "gti" in data
        assert "mjdref" in data
        gtis = data["gti"]
        lengths = np.array([g1 - g0 for (g0, g1) in gtis])
        assert np.all(lengths <= 10)

    def test_split_events(self):
        treat_event_file(self.fits_fileA)

        filea = os.path.join(
            self.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )

        files = hen.read_events.main_splitevents([filea, "-l", "50"])
        for f in files:
            assert os.path.exists(f)

    def test_load_events(self):
        """Test event file reading."""
        command = "{}".format(self.fits_fileA)
        hen.read_events.main(command.split())
        ev = hen.io.load_events(self.ev_fileA)
        assert hasattr(ev, "header")
        assert hasattr(ev, "gti")

    def test_load_events_with_2_cpus(self):
        """Test event file reading."""
        command = "{} {} --nproc 2".format(
            os.path.join(self.datadir, "monol_testB.evt"),
            os.path.join(self.datadir, "monol_testA_timezero.evt"),
        )
        hen.read_events.main(command.split())

    def test_load_events_split(self):
        """Test event file splitting."""
        command = "{0} -g --min-length 0".format(self.fits_fileB)
        hen.read_events.main(command.split())
        new_filename = os.path.join(
            self.datadir,
            "monol_testB_nustar_fpmb_gti000_ev" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(new_filename)
        command = "{0}".format(new_filename)
        hen.lcurve.main(command.split())
        new_filename = os.path.join(
            self.datadir,
            "monol_testB_nustar_fpmb_gti000_lc" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(new_filename)
        lc = hen.io.load_lcurve(new_filename)
        gti_to_test = hen.io.load_events(self.ev_fileB).gti[0]
        assert np.allclose(gti_to_test, lc.gti)

    def test_load_gtis(self):
        """Test loading of GTIs from FITS files."""
        fits_file = os.path.join(self.datadir, "monol_testA.evt")
        hen.io.load_gtis(fits_file)

    def test_load_events_noclobber(self):
        """Test event file reading w. noclobber option."""
        with catch_warnings() as w:
            command = "{0} --noclobber".format(self.fits_fileB)
            hen.read_events.main(command.split())
        assert (
            str(w[0].message)
            .strip()
            .endswith("exists and using noclobber. Skipping")
        ), "Unexpected warning output"

    @classmethod
    def teardown_class(cls):
        for pattern in [
            "monol_*" + HEN_FILE_EXTENSION,
            "*phasetag*",
            "*fake*",
            "monol*.pdf",
        ]:
            files = glob.glob(os.path.join(cls.datadir, pattern))
            for file in files:
                os.unlink(file)
