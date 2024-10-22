# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os

import numpy as np
import pytest
from stingray.lightcurve import Lightcurve

from astropy import log
from astropy.logger import AstropyUserWarning
from hendrics.io import HEN_FILE_EXTENSION, get_file_type
from hendrics.lcurve import lcurve_from_events
from hendrics.read_events import treat_event_file
from hendrics.tests import _dummy_par

from hendrics import (
    base,
    calibrate,
    create_gti,
    exvar,
    fspec,
    io,
    plot,
    rebin,
    read_events,
    lcurve,
    exposure,
)
from . import cleanup_test_dir

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

log.setLevel("DEBUG")


class TestLcurve:
    """Real unit tests."""

    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        cls.fits_fileA = os.path.join(cls.datadir, "monol_testA.evt")
        cls.new_filename = os.path.join(
            cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
        )
        cls.calib_filename = os.path.join(
            cls.datadir,
            "monol_testA_nustar_fpma_ev_calib" + HEN_FILE_EXTENSION,
        )

    def test_treat_event_file_nustar(self):
        from astropy.io.fits import Header

        treat_event_file(self.fits_fileA, discard_calibration=True)
        lcurve_from_events(self.new_filename)
        newfile = os.path.join(self.datadir, "monol_testA_nustar_fpma_lc" + HEN_FILE_EXTENSION)
        assert os.path.exists(newfile)
        type, data = get_file_type(newfile)
        assert type == "lc"
        assert isinstance(data, Lightcurve)
        Header.fromstring(data.header)
        assert hasattr(data, "mjdref")
        assert data.mjdref > 0

    def test_treat_event_file_nustar_energy(self):
        rmf = os.path.join(self.datadir, "test.rmf")
        command = f"{self.new_filename} -r {rmf} --nproc 2"
        calibrate.main(command.split())
        lcurve_from_events(self.calib_filename, e_interval=[3, 50])

        newfile = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_E3-50_lc" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(newfile)
        type, data = get_file_type(newfile)
        assert type == "lc"
        assert isinstance(data, Lightcurve)
        assert hasattr(data, "mjdref")
        assert data.mjdref > 0

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")


class TestFullRun:
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
        data_a, data_b = (
            os.path.join(cls.datadir, "monol_testA.evt"),
            os.path.join(cls.datadir, "monol_testB.evt"),
        )
        command = f"{data_a} {data_b} --discard-calibration"
        read_events.main(command.split())
        data_a, data_b, rmf = (
            os.path.join(cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir, "test.rmf"),
        )
        command = f"{data_a} {data_b} -r {rmf}"
        calibrate.main(command.split())

    def test_lcurve(self):
        """Test light curve production."""
        from astropy.io.fits import Header

        new_filename = os.path.join(
            os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        )
        command = (
            f"{self.ev_fileAcal} -e {3} {50} --safe-interval "
            f"{100} {300}  --nproc 2 -b 0.5 -o {new_filename}"
        )
        lcurve.main(command.split())

        assert os.path.exists(new_filename)
        lc = io.load_lcurve(new_filename)
        assert hasattr(lc, "header")
        # Test that the header is correctly conserved
        Header.fromstring(lc.header)
        assert hasattr(lc, "gti")
        gti_to_test = io.load_events(self.ev_fileAcal).gti
        assert np.allclose(gti_to_test, lc.gti)

    def test_lcurve_B(self):
        out = os.path.join(self.datadir, "monol_testB_E3-50_lc" + HEN_FILE_EXTENSION)
        command = f"{self.ev_fileBcal} -e 3 50 --safe-interval 100 300 -b 0.5 -o {out}"
        lcurve.main(command.split())
        assert os.path.exists(
            os.path.join(self.datadir, "monol_testB_E3-50_lc" + HEN_FILE_EXTENSION)
        )

    def test_weight_lcurve(self):
        """Test light curve production."""
        from astropy.io.fits import Header

        new_lc = "polar_weightbla_lc" + HEN_FILE_EXTENSION
        new_ev = "polar_ev" + HEN_FILE_EXTENSION

        events = io.load_events(self.ev_fileAcal)
        events.bla = np.random.uniform(0, 1, events.time.size)

        io.save_events(events, new_ev)

        command = f"{new_ev} -b 10 --weight-on bla"
        lcurve.main(command.split())

        assert os.path.exists(new_lc)
        lc = io.load_lcurve(new_lc)
        assert hasattr(lc, "header")

        # Test that the header is correctly conserved
        Header.fromstring(lc.header)
        assert hasattr(lc, "gti")
        gti_to_test = io.load_events(new_ev).gti
        assert np.allclose(gti_to_test, lc.gti)
        assert lc.err_dist == "gauss"

    def test_lcurve_noclobber(self):
        input_file = self.ev_fileAcal
        new_filename = os.path.join(
            os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        )

        with pytest.warns(AstropyUserWarning, match="File exists, and noclobber"):
            command = f"{input_file} -o {new_filename} --noclobber"
            lcurve.main(command.split())

    def test_lcurve_split(self):
        """Test lc with gti-split option."""
        command = f"{self.ev_fileAcal} {self.ev_fileBcal} -g"
        lcurve.main(command.split())
        new_filename = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_gti000_lc" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(new_filename)
        lc = io.load_lcurve(new_filename)
        gti_to_test = io.load_events(self.ev_fileAcal).gti[0]
        assert np.allclose(gti_to_test, lc.gti)

    def test_fits_lcurve0(self):
        """Test light curves from FITS."""
        lcurve_ftools_orig = os.path.join(self.datadir, "lcurveA.fits")

        lcurve_ftools = os.path.join(self.datadir, "lcurve_ftools_lc" + HEN_FILE_EXTENSION)

        out = os.path.join(self.datadir, "lcurve_lc")
        command = f"{self.ev_fileAcal} --outfile {out}"

        lcurve.main(command.split())
        assert os.path.exists(os.path.join(self.datadir, "lcurve_lc") + HEN_FILE_EXTENSION)

        command = f"--fits-input {lcurve_ftools_orig} --outfile {lcurve_ftools}"
        lcurve.main(command.split())
        with pytest.warns(AstropyUserWarning, match="File exists, and noclobber"):
            command = command + " --noclobber"
            lcurve.main(command.split())

    def test_fits_lcurve1(self):
        """Test light curves from FITS."""
        lcurve_ftools = os.path.join(self.datadir, "lcurve_ftools_lc" + HEN_FILE_EXTENSION)

        lcurve_mp = os.path.join(self.datadir, "lcurve_lc" + HEN_FILE_EXTENSION)

        _, lcdata_mp = io.get_file_type(lcurve_mp, raw_data=False)
        _, lcdata_ftools = io.get_file_type(lcurve_ftools, raw_data=False)

        lc_mp = lcdata_mp.counts

        lenmp = len(lc_mp)
        lc_ftools = lcdata_ftools.counts
        lenftools = len(lc_ftools)
        goodlen = min([lenftools, lenmp])

        diff = lc_mp[:goodlen] - lc_ftools[:goodlen]

        assert np.all(np.abs(diff) <= 1e-3), "Light curve data do not coincide between FITS and HEN"

    def test_txt_lcurve(self):
        """Test light curves from txt."""
        lcurve_mp = os.path.join(self.datadir, "lcurve_lc" + HEN_FILE_EXTENSION)
        _, lcdata_mp = io.get_file_type(lcurve_mp, raw_data=False)
        lc_mp = lcdata_mp.counts
        time_mp = lcdata_mp.time

        lcurve_txt_orig = os.path.join(self.datadir, "lcurve_txt_lc.txt")

        io.save_as_ascii([time_mp, lc_mp], lcurve_txt_orig)

        lcurve_txt = os.path.join(self.datadir, "lcurve_txt_lc" + HEN_FILE_EXTENSION)
        command = "--txt-input " + lcurve_txt_orig + " --outfile " + lcurve_txt
        lcurve.main(command.split())
        lcdata_txt = io.get_file_type(lcurve_txt, raw_data=False)[1]

        lc_txt = lcdata_txt.counts

        assert np.all(
            np.abs(lc_mp - lc_txt) <= 1e-3
        ), "Light curve data do not coincide between txt and HEN"

        with pytest.warns(AstropyUserWarning, match="File exists, and noclobber"):
            command = command + " --noclobber"
            lcurve.main(command.split())

    def test_joinlcs(self):
        """Test produce joined light curves."""
        new_filename = os.path.join(self.datadir, "monol_test_joinlc" + HEN_FILE_EXTENSION)
        # because join_lightcurves separates by instrument
        new_actual_filename = os.path.join(
            self.datadir, "fpmamonol_test_joinlc" + HEN_FILE_EXTENSION
        )
        lcA_pattern = "monol_testA_nustar_fpma_gti[0-9][0-9][0-9]_lc*"
        lcB_pattern = "monol_testB_nustar_fpmb_gti[0-9][0-9][0-9]_lc*"
        lcurve.join_lightcurves(
            glob.glob(os.path.join(self.datadir, lcA_pattern + HEN_FILE_EXTENSION))
            + glob.glob(os.path.join(self.datadir, lcB_pattern + HEN_FILE_EXTENSION)),
            new_filename,
        )

        lc = io.load_lcurve(new_actual_filename)
        assert hasattr(lc, "gti")
        gti_to_test = io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, lc.gti)

    def test_scrunchlcs(self):
        """Test produce scrunched light curves."""
        a_in = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        b_in = os.path.join(self.datadir, "monol_testB_E3-50_lc" + HEN_FILE_EXTENSION)
        out = os.path.join(self.datadir, "monol_test_scrunchlc" + HEN_FILE_EXTENSION)
        command = f"{a_in} {b_in} -o {out}"

        a_lc = io.load_lcurve(a_in)
        b_lc = io.load_lcurve(b_in)
        a_lc.apply_gtis()
        b_lc.apply_gtis()
        lcurve.scrunch_main(command.split())
        out_lc = io.load_lcurve(out)
        out_lc.apply_gtis()
        assert np.all(out_lc.counts == a_lc.counts + b_lc.counts)
        gti_to_test = io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def testbaselinelc(self):
        """Test produce scrunched light curves."""
        a_in = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        out = os.path.join(self.datadir, "monol_test_baselc")
        command = f"{a_in} -o {out} -p 0.001 --lam 1e5"

        lcurve.baseline_main(command.split())
        out_lc = io.load_lcurve(out + "_0" + HEN_FILE_EXTENSION)
        assert hasattr(out_lc, "base")
        gti_to_test = io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def testbaselinelc_nooutroot(self):
        """Test produce scrunched light curves."""
        a_in = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        command = f"{a_in} -p 0.001 --lam 1e5"

        lcurve.baseline_main(command.split())
        out_lc = io.load_lcurve(base.hen_root(a_in) + "_lc_baseline" + HEN_FILE_EXTENSION)
        assert hasattr(out_lc, "base")
        gti_to_test = io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def test_lcurve_error_uncalibrated(self):
        """Test light curve error from uncalibrated file."""
        data = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION,
        )
        command = f"{data} -e 3 50"

        with pytest.raises(ValueError, match="Did you run HENcalibrate?"):
            lcurve.main(command.split())

    def test_lcurve_pi_filtering(self):
        """Test light curve using PI filtering."""
        data = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION,
        )
        command = f"{data} --pi-interval {10} {300}"

        lcurve.main(command.split())

    def test_rebinlc(self):
        """Test LC rebinning."""
        data = os.path.join(self.datadir, "monol_testA_E3-50_lc") + HEN_FILE_EXTENSION
        command = f"{data} -r 4"
        rebin.main(command.split())

    def test_save_fvar_from_lc(self):
        fname = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        exvar.main([fname, "-c", "10", "--fraction-step", "0.6", "--norm", "fvar"])
        out = base.hen_root(fname) + "_fvar" + ".qdp"
        os.path.exists(out)

    def test_save_excvar_from_lc(self):
        fname = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        exvar.main([fname])
        out = base.hen_root(fname) + "_excvar" + ".qdp"
        os.path.exists(out)

    def test_save_excvar_norm_from_lc(self):
        fname = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        exvar.main([fname, "--norm", "norm_excvar"])
        out = base.hen_root(fname) + "_norm_excvar" + ".qdp"
        os.path.exists(out)

    def test_save_excvar_wrong_norm_from_lc(self):
        fname = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        with pytest.raises(ValueError, match="Normalization must be fvar,"):
            exvar.main([fname, "--norm", "cicciput"])

    def test_create_gti_lc(self):
        """Test creating a GTI file."""
        fname = os.path.join(self.datadir, "monol_testA_E3-50_lc") + HEN_FILE_EXTENSION
        command = f"{fname} -f counts>0 -c --debug"
        create_gti.main(command.split())

    def test_apply_gti_lc(self):
        """Test applying a GTI file."""
        fname = os.path.join(self.datadir, "monol_testA_E3-50_gti") + HEN_FILE_EXTENSION
        lcfname = os.path.join(self.datadir, "monol_testA_E3-50_lc") + HEN_FILE_EXTENSION
        lcoutname = os.path.join(self.datadir, "monol_testA_E3-50_lc_gtifilt") + HEN_FILE_EXTENSION
        command = f"{lcfname} -a {fname} --debug"
        create_gti.main(command.split())
        io.load_lcurve(lcoutname)

    def test_plot_lcurve_baseline(self):
        a_in = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        base_file = base.hen_root(a_in) + "_lc_baseline" + HEN_FILE_EXTENSION
        plot.main([base_file, "--noplot", "-o", "dummy_base.qdp"])
        filedata = np.genfromtxt("dummy_base.qdp")

        assert filedata.shape[1] == 3

    def test_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(self.datadir, "lcurve_ftools_lc" + HEN_FILE_EXTENSION)
        command = f"{lcurve_ftools} --save-all -f 128"
        fspec.main(command.split())

    def test_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(self.datadir, "lcurve_txt_lc" + HEN_FILE_EXTENSION)
        command = f"{lcurve_txt} --save-all -f 128"
        fspec.main(command.split())

    def test_exposure(self):
        """Test exposure calculations from unfiltered files."""
        lcname = os.path.join(self.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION)
        ufname = os.path.join(self.datadir, "monol_testA_uf.evt")
        command = f"{lcname} {ufname}"

        exposure.main(command.split())
        fname = os.path.join(self.datadir, "monol_testA_E3-50_lccorr" + HEN_FILE_EXTENSION)
        assert os.path.exists(fname)
        ftype, contents = io.get_file_type(fname)

        assert isinstance(contents, Lightcurve)
        assert hasattr(contents, "expo")

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")
