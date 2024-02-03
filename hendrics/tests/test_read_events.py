# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import glob
import pytest
import numpy as np

from stingray.events import EventList
from hendrics.read_events import treat_event_file
from hendrics.io import (
    HEN_FILE_EXTENSION,
    load_data,
    save_events,
    load_events,
)
from hendrics.io import ref_mjd
from hendrics.io import main as main_readfile
from hendrics.fake import main
import hendrics as hen
from . import cleanup_test_dir


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
        with pytest.warns(UserWarning, match="changing MJDREF"):
            hen.read_events.main_join(
                [
                    self.f0,
                    self.f1,
                    "-o",
                    os.path.join(self.datadir, "monol_merg_ev" + HEN_FILE_EXTENSION),
                ]
            )

        out = os.path.join(self.datadir, "monol_merg_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)
        os.unlink(out)

    def test_merge_events_different_instr(self):
        with pytest.warns(UserWarning):
            hen.read_events.main_join(
                [
                    self.f0,
                    self.f3,
                    "-o",
                    os.path.join(self.datadir, "monol_merg13_ev" + HEN_FILE_EXTENSION),
                ]
            )

        out = os.path.join(self.datadir, "monol_merg13_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)
        os.unlink(out)

    def test_merge_events_different_instr_ignore(self):
        hen.read_events.main_join(
            [
                self.f0,
                self.f2,
                self.f3,
                "--ignore-instr",
                "-o",
                os.path.join(self.datadir, "monol_merg1023_ev" + HEN_FILE_EXTENSION),
            ]
        )

        out = os.path.join(self.datadir, "monol_merg1023_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)
        ev = load_events(out)
        assert ev.instr.lower() == "BA,BA,BU".lower()
        os.unlink(out)

    def test_merge_two_events_different_instr_ignore(self):
        hen.read_events.main_join(
            [
                self.f0,
                self.f3,
                "--ignore-instr",
                "-o",
                os.path.join(self.datadir, "monol_merg13_ev" + HEN_FILE_EXTENSION),
            ]
        )

        out = os.path.join(self.datadir, "monol_merg13_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)
        ev = load_events(out)
        assert ev.instr.lower() == "BA,BU".lower()
        os.unlink(out)

    def test_merge_events_no_out_fname(self):
        with pytest.warns(UserWarning, match="changing MJDREF"):
            hen.read_events.main_join([self.f0, self.f1])
        out = os.path.join(self.datadir, "ev_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)
        os.unlink(out)

    def test_merge_many_events_warnings(self):
        out = os.path.join(self.datadir, "monol_merg_many_ev" + HEN_FILE_EXTENSION)
        with pytest.warns(
            UserWarning,
            match=f"{os.path.split(self.f1)[1]}.* has a different MJDREF",
        ):
            hen.read_events.main_join([self.f0, self.f1, self.f2, "-o", out])
        assert os.path.exists(out)
        os.unlink(out)
        with pytest.warns(
            UserWarning,
            match=f"{os.path.split(self.f3)[1]}.* is from a different",
        ):
            hen.read_events.main_join([self.f0, self.f2, self.f3, "-o", out])
        assert os.path.exists(out)
        os.unlink(out)

        with pytest.warns(
            UserWarning,
            match=f"{os.path.split(self.f4)[1]}.* has no good events",
        ):
            hen.read_events.main_join([self.f0, self.f2, self.f4, "-o", out])
        assert os.path.exists(out)
        os.unlink(out)

    def test_merge_many_events(self):
        outfile = "joint_ev" + HEN_FILE_EXTENSION
        # Note that only 0 and 2 are valid
        with pytest.warns(UserWarning):
            hen.read_events.main_join([self.f0, self.f2, self.f3])

        assert os.path.exists(outfile)

        data = load_events(outfile)
        assert hasattr(data, "gti")
        assert data.gti is not None
        allgti = []
        # Note that only 0 and 2 are valid
        for evfile in [self.f0, self.f2]:
            ev = load_events(evfile)
            allgti.append(ev.gti)
        allgti = np.sort(np.concatenate(allgti))
        assert np.allclose(data.gti, allgti)
        os.unlink(outfile)

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")


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

    def test_start(self):
        """Make any warnings in setup_class be dumped here."""
        pass

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
        assert "pi" in data and data["pi"].size > 0

    def test_treat_event_file_xmm_gtisplit(self):
        treat_event_file(self.fits_file, gti_split=True)
        new_filename = "monol_test_fake_xmm_epn_det01_gti000_ev" + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir, new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert "instr" in data
        assert "gti" in data
        assert "mjdref" in data

    def test_treat_event_file_xmm_lensplit(self):
        treat_event_file(self.fits_file, length_split=100)
        new_filename = "monol_test_fake_xmm_epn_det01_chunk000_ev" + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir, new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert "instr" in data
        assert "gti" in data
        assert "mjdref" in data
        gti = data["gti"]
        lengths = np.array([g1 - g0 for (g0, g1) in gti])
        # add an epsilon for numerical error
        assert np.all(lengths <= 100 + 1e-7)

    def test_split_events(self):
        treat_event_file(self.fits_fileA)

        filea = os.path.join(
            self.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )

        files = hen.read_events.main_splitevents([filea, "-l", "50"])
        for f in files:
            assert os.path.exists(f)

    def test_split_events_at_mjd(self):
        treat_event_file(self.fits_fileA)

        filea = os.path.join(
            self.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )
        data = load_events(filea)
        mean_met = np.mean(data.time)
        mean_mjd = mean_met / 86400 + data.mjdref

        files = hen.read_events.main_splitevents(
            [filea, "--split-at-mjd", f"{mean_mjd}"]
        )
        assert "before" in files[0]
        assert "after" in files[1]

        data = load_events(files[0])
        assert np.isclose(data.gti[-1, 1], mean_met)
        data = load_events(files[1])
        assert np.isclose(data.gti[0, 0], mean_met)

    def test_split_events_bad_overlap_raises(self):
        treat_event_file(self.fits_fileA)

        filea = os.path.join(
            self.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )

        with pytest.raises(ValueError, match="Overlap cannot be >=1. Exiting."):
            hen.read_events.split_eventlist(filea, 10, overlap=1.5)

    def test_load_events(self):
        """Test event file reading."""
        command = "{}".format(self.fits_fileA)
        hen.read_events.main(command.split())
        ev = hen.io.load_events(self.ev_fileA)
        assert hasattr(ev, "header")
        assert hasattr(ev, "gti")

        main_readfile([self.ev_fileA])

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

    def test_load_events_noclobber(self):
        """Test event file reading w. noclobber option."""
        with pytest.warns(UserWarning, match="exists and using noclobber. Skipping"):
            command = "{0} --noclobber".format(self.fits_fileB)
            hen.read_events.main(command.split())

    def test_fix_gaps_events(self):
        """Test event file reading w. noclobber option."""
        command = "{0} --fill-small-gaps 4".format(self.fits_fileB)
        hen.read_events.main(command.split())

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
