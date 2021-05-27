import shutil
import os
import glob
import subprocess as sp

import numpy as np
import stingray
from astropy import log
import pytest
from stingray.events import EventList
import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics import (
    calibrate,
    colors,
    io,
    lcurve,
    modeling,
    plot,
    read_events,
    rebin,
    save_as_xspec,
    sum_fspec,
    timelags,
    varenergy,
)

from hendrics.base import touch
from hendrics.fspec import calc_cpds, calc_pds

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION

log.setLevel("DEBUG")
# log.basicConfig(filename='HEN.log', level=log.DEBUG, filemode='w')


def test_pds_fails_noclobber_exists():
    touch("bububu")
    with pytest.warns(UserWarning) as record:
        calc_pds("bla.p", 512, outname="bububu", noclobber=True)
    assert np.any(
        ["File exists, and noclobber" in r.message.args[0] for r in record]
    )
    os.unlink("bububu")


def test_cpds_fails_noclobber_exists():
    touch("bububu")
    with pytest.warns(UserWarning) as record:
        calc_cpds("bla.p", "blu.p", 512, outname="bububu", noclobber=True)
    assert np.any(
        ["File exists, and noclobber" in r.message.args[0] for r in record]
    )
    os.unlink("bububu")


def test_distributed_pds():
    events = EventList(
        np.sort(np.random.uniform(0, 1000, 1000)), gti=[[0, 1000]]
    )
    single_periodogram = stingray.AveragedPowerspectrum(
        events, segment_size=100, dt=0.1, norm="leahy"
    )
    pds_iterable = hen.fspec._provide_periodograms(events, 100, 0.1, "leahy")
    pds_distr = hen.fspec.average_periodograms(pds_iterable)
    assert np.allclose(pds_distr.power, single_periodogram.power)
    assert np.allclose(pds_distr.power_err, single_periodogram.power_err)
    assert np.allclose(pds_distr.freq, single_periodogram.freq)
    assert pds_distr.m == single_periodogram.m


def test_distributed_cpds():
    events1 = EventList(
        np.sort(np.random.uniform(0, 1000, 1000)), gti=[[0, 1000]]
    )
    events2 = EventList(
        np.sort(np.random.uniform(0, 1000, 1000)), gti=[[0, 1000]]
    )
    single_periodogram = stingray.AveragedCrossspectrum(
        events1, events2, segment_size=100, dt=0.1, norm="leahy"
    )
    pds_iterable = hen.fspec._provide_cross_periodograms(
        events1, events2, 100, 0.1, "leahy"
    )
    pds_distr = hen.fspec.average_periodograms(pds_iterable)
    assert np.allclose(pds_distr.power, single_periodogram.power)
    assert np.allclose(pds_distr.power_err, single_periodogram.power_err)
    assert np.allclose(pds_distr.freq, single_periodogram.freq)
    assert pds_distr.m == single_periodogram.m


class TestFullRun(object):
    """Test how command lines work.

    Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/python-unittest-testcase-execution-order

    When command line is missing, uses some function calls
    """  # NOQA

    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")

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
            os.path.join(
                cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION
            ),
            os.path.join(
                cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION
            ),
            os.path.join(cls.datadir, "test.rmf"),
        )
        hen.calibrate.main(command.split())
        cls.lcA = os.path.join(
            os.path.join(
                cls.datadir, "monol_testA_nustar_fpma_lc" + HEN_FILE_EXTENSION
            )
        )
        cls.lcB = os.path.join(
            os.path.join(
                cls.datadir, "monol_testB_nustar_fpmb_lc" + HEN_FILE_EXTENSION
            )
        )
        cls.lcAfilt = os.path.join(
            os.path.join(
                cls.datadir, "monol_testA_E3-50_lc" + HEN_FILE_EXTENSION
            )
        )
        cls.lcBfilt = os.path.join(
            os.path.join(
                cls.datadir, "monol_testB_E3-50_lc" + HEN_FILE_EXTENSION
            )
        )
        command = (
            "{} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " "-o {}"
        ).format(cls.ev_fileAcal, cls.lcAfilt)
        hen.lcurve.main(command.split())
        command = (
            "{} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 " "-o {}"
        ).format(cls.ev_fileBcal, cls.lcBfilt)
        hen.lcurve.main(command.split())
        command = ("{} --nproc 2 -b 0.5 " "-o {}").format(
            cls.ev_fileAcal, cls.lcA
        )
        hen.lcurve.main(command.split())
        command = ("{}  --nproc 2 -b 0.5 " "-o {}").format(
            cls.ev_fileBcal, cls.lcB
        )
        hen.lcurve.main(command.split())

        cls.pdsA = os.path.join(
            cls.datadir, "monol_testA_E3-50_pds" + HEN_FILE_EXTENSION
        )
        cls.pdsB = os.path.join(
            cls.datadir, "monol_testB_E3-50_pds" + HEN_FILE_EXTENSION
        )
        cls.cpds = os.path.join(
            cls.datadir, "monol_test_E3-50_cpds" + HEN_FILE_EXTENSION
        )

        command = "{} {} -f 128 -k PDS --save-all --norm leahy".format(
            cls.lcAfilt, cls.lcBfilt
        )
        hen.fspec.main(command.split())

        command = "{} {} -f 128 -k CPDS --save-all --norm leahy".format(
            cls.lcAfilt, cls.lcBfilt
        )
        hen.fspec.main(command.split())
        assert os.path.exists(cls.cpds)
        assert os.path.exists(cls.pdsA)
        assert os.path.exists(cls.pdsB)

    def test_pds_leahy_dtbig(self):
        """Test PDS production."""
        lc = self.lcAfilt
        hen.io.main([lc])
        command = "{0} -f 128 -k PDS --save-all --norm leahy -b {1}".format(
            lc, 1
        )
        hen.fspec.main(command.split())

        assert os.path.exists(
            os.path.join(
                self.datadir, "monol_testA_E3-50_pds" + HEN_FILE_EXTENSION
            )
        )

    def test_pds_leahy(self):
        """Test PDS production."""
        evdata = self.ev_fileA
        lcdata = self.lcA

        command = "{0} -f 128 -k PDS --norm leahy -b 0.5".format(evdata)
        hen.fspec.main(command.split())
        command = "{0} -f 128 -k PDS --save-all --norm leahy".format(lcdata)
        hen.fspec.main(command.split())

        evout = evdata.replace("_ev", "_pds")
        lcout = lcdata.replace("_lc", "_pds")
        assert os.path.exists(evout)
        assert os.path.exists(lcout)
        evpds = hen.io.load_pds(evout)
        lcpds = hen.io.load_pds(lcout)
        assert np.allclose(evpds.power, lcpds.power)

    @pytest.mark.parametrize("data_kind", ["events", "lc"])
    def test_pds(self, data_kind):
        """Test PDS production."""
        if data_kind == "events":
            labelA = "nustar_fpma_ev"
            labelB = "nustar_fpmb_ev"
        else:
            labelA = labelB = "E3-50_lc"
        command = (
            "{0} {1} -f 128 --save-all --save-dyn -k PDS "
            "--norm frac".format(
                os.path.join(self.datadir, f"monol_testA_{labelA}")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, f"monol_testB_{labelB}")
                + HEN_FILE_EXTENSION,
            )
        )
        hen.fspec.main(command.split())

        out_labelA = labelA.replace("_ev", "_pds").replace("_lc", "_pds")
        out_labelB = labelB.replace("_ev", "_pds").replace("_lc", "_pds")
        assert os.path.exists(
            os.path.join(
                self.datadir, f"monol_testB_{out_labelB}" + HEN_FILE_EXTENSION
            )
        )
        assert os.path.exists(
            os.path.join(self.datadir, f"monol_testA_{out_labelA}")
            + HEN_FILE_EXTENSION
        )

    @pytest.mark.parametrize("kind", ["PDS", "CPDS"])
    def test_pds_events_big(self, kind):
        """Test PDS production."""
        labelA = "nustar_fpma_ev"
        labelB = "nustar_fpmb_ev"

        command = (
            "{0} {1} -f 128 --save-all --save-dyn -k {2} "
            "--norm frac --test".format(
                os.path.join(self.datadir, f"monol_testA_{labelA}")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, f"monol_testB_{labelB}")
                + HEN_FILE_EXTENSION,
                kind,
            )
        )
        hen.fspec.main(command.split())

    def test_cpds_ignore_instr(self):
        """Test CPDS production."""
        command = (
            "{0} {1} -f 128 --save-dyn -k CPDS,lag --save-all --ignore-instr"
            " -o {2} --debug".format(
                os.path.join(self.datadir, "monol_testA_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_testB_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "ignore_instr")
                + HEN_FILE_EXTENSION,
            )
        )

        hen.fspec.main(command.split())

    def test_cpds_rms_norm(self):
        """Test CPDS production."""
        command = (
            "{0} {1} -f 128 --save-dyn -k CPDS --save-all "
            "--norm rms -o {2}".format(
                os.path.join(self.datadir, "monol_testA_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_testB_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_test_E3-50"),
            )
        )

        hen.fspec.main(command.split())

    def test_cpds_wrong_norm(self):
        """Test CPDS production."""
        command = (
            "{0} {1} -f 128 --save-dyn -k CPDS --norm blablabla -o {2}".format(
                os.path.join(self.datadir, "monol_testA_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_testB_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_test_E3-50"),
            )
        )
        with pytest.warns(UserWarning) as record:
            hen.fspec.main(command.split())

        assert np.any(
            [
                "Beware! Unknown normalization" in r.message.args[0]
                for r in record
            ]
        )

    def test_cpds_dtbig(self):
        """Test CPDS production."""
        command = (
            "{0} {1} -f 128 --save-dyn -k CPDS --save-all --norm "
            "frac -o {2}".format(
                os.path.join(self.datadir, "monol_testA_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_testB_E3-50_lc")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_test_E3-50"),
            )
        )
        command += " -b 1"
        hen.fspec.main(command.split())

    def test_dumpdynpds(self):
        """Test dump dynamical PDSs."""
        command = (
            "--noplot "
            + os.path.join(self.datadir, "monol_testA_E3-50_pds")
            + HEN_FILE_EXTENSION
        )
        with pytest.raises(NotImplementedError):
            hen.fspec.dumpdyn_main(command.split())

    def test_sumpds(self):
        """Test the sum of pdss."""
        hen.sum_fspec.main(
            [
                os.path.join(self.datadir, "monol_testA_E3-50_pds")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_testB_E3-50_pds")
                + HEN_FILE_EXTENSION,
                "-o",
                os.path.join(
                    self.datadir, "monol_test_sum" + HEN_FILE_EXTENSION
                ),
            ]
        )

    def test_dumpdyncpds(self):
        """Test dump dynamical PDSs."""
        command = (
            "--noplot "
            + os.path.join(self.datadir, "monol_test_E3-50_cpds")
            + HEN_FILE_EXTENSION
        )
        with pytest.raises(NotImplementedError):
            hen.fspec.dumpdyn_main(command.split())

    def test_rebinpds(self):
        """Test PDS rebinning 1."""
        command = "{0} -r 2".format(
            os.path.join(self.datadir, "monol_testA_E3-50_pds")
            + HEN_FILE_EXTENSION
        )
        hen.rebin.main(command.split())
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_testA_E3-50_pds_rebin2" + HEN_FILE_EXTENSION,
            )
        )

    def test_rebinpds_geom(self):
        """Test geometrical PDS rebinning."""
        command = "{0} {1} -r 1.03".format(
            os.path.join(self.datadir, "monol_testA_E3-50_pds")
            + HEN_FILE_EXTENSION,
            os.path.join(self.datadir, "monol_testB_E3-50_pds")
            + HEN_FILE_EXTENSION,
        )
        hen.rebin.main(command.split())
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_testA_E3-50_pds_rebin1.03" + HEN_FILE_EXTENSION,
            )
        )
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_testB_E3-50_pds_rebin1.03" + HEN_FILE_EXTENSION,
            )
        )

    def test_rebincpds(self):
        """Test CPDS rebinning."""
        command = "{0} -r 2".format(
            os.path.join(self.datadir, "monol_test_E3-50_cpds")
            + HEN_FILE_EXTENSION
        )
        hen.rebin.main(command.split())
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_test_E3-50_cpds_rebin2" + HEN_FILE_EXTENSION,
            )
        )

    def test_rebincpds_geom(self):
        """Test CPDS geometrical rebinning."""
        command = "{0} -r 1.03".format(
            os.path.join(self.datadir, "monol_test_E3-50_cpds")
            + HEN_FILE_EXTENSION
        )
        hen.rebin.main(command.split())
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_test_E3-50_cpds_rebin1.03" + HEN_FILE_EXTENSION,
            )
        )

    def test_save_lags(self):
        fname = os.path.join(
            self.datadir, "monol_test_E3-50_cpds_rebin2" + HEN_FILE_EXTENSION
        )
        hen.timelags.main([fname])
        out = hen.base.hen_root(fname) + "_lags.qdp"
        os.path.exists(out)

    def test_fit_pds(self):
        modelstring = """
from astropy.modeling import models
model = models.Const1D()
        """
        modelfile = "bubu__model__.py"
        with open(modelfile, "w") as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = os.path.join(
            self.datadir, "monol_testA_E3-50_pds" + HEN_FILE_EXTENSION
        )
        pdsfile2 = os.path.join(
            self.datadir, "monol_testB_E3-50_pds" + HEN_FILE_EXTENSION
        )

        command = "{0} {1} -m {2} --frequency-interval 0 10".format(
            pdsfile1, pdsfile2, modelfile
        )
        hen.modeling.main_model(command.split())

        out0 = os.path.join(self.datadir, "monol_testA_E3-50_pds_bestfit.p")
        out1 = os.path.join(self.datadir, "monol_testB_E3-50_pds_bestfit.p")
        assert os.path.exists(out0)
        assert os.path.exists(out1)
        m, k, c = hen.io.load_model(
            os.path.join(self.datadir, "monol_testB_E3-50_pds_bestfit.p")
        )
        assert hasattr(m, "amplitude")
        os.unlink(out0)
        os.unlink(out1)

        out0 = os.path.join(
            self.datadir, "monol_testA_E3-50_pds_fit" + HEN_FILE_EXTENSION
        )
        out1 = os.path.join(
            self.datadir, "monol_testB_E3-50_pds_fit" + HEN_FILE_EXTENSION
        )
        assert os.path.exists(out0)
        assert os.path.exists(out1)
        spec = hen.io.load_pds(out0)
        assert hasattr(spec, "best_fits")

    def test_fit_cpds(self):
        modelstring = """
from astropy.modeling import models
model = models.Const1D()
        """
        modelfile = "bubu__model__.py"
        with open(modelfile, "w") as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = os.path.join(
            self.datadir, "monol_test_E3-50_cpds" + HEN_FILE_EXTENSION
        )

        command = "{0} -m {1} --frequency-interval 0 10".format(
            pdsfile1, modelfile
        )
        hen.modeling.main_model(command.split())

        out0 = os.path.join(self.datadir, "monol_test_E3-50_cpds_bestfit.p")
        assert os.path.exists(out0)
        m, k, c = hen.io.load_model(out0)
        assert hasattr(m, "amplitude")
        os.unlink(out0)

        out0 = os.path.join(
            self.datadir, "monol_test_E3-50_cpds_fit" + HEN_FILE_EXTENSION
        )
        assert os.path.exists(out0)
        spec = hen.io.load_pds(out0)
        assert hasattr(spec, "best_fits")

    def test_fit_pds_f_no_of_intervals_invalid(self):
        modelstring = """
from astropy.modeling import models
model = models.Const1D()
        """
        modelfile = "bubu__model__.py"
        with open(modelfile, "w") as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = os.path.join(
            self.datadir, "monol_testA_E3-50_pds" + HEN_FILE_EXTENSION
        )
        pdsfile2 = os.path.join(
            self.datadir, "monol_testB_E3-50_pds" + HEN_FILE_EXTENSION
        )

        command = "{0} {1} -m {2} --frequency-interval 0 1 9".format(
            pdsfile1, pdsfile2, modelfile
        )
        with pytest.raises(ValueError) as excinfo:
            hen.modeling.main_model(command.split())

        assert "Invalid number of frequencies specified" in str(excinfo.value)

    def test_savexspec(self):
        """Test save as Xspec 1."""
        command = "{0}".format(
            os.path.join(self.datadir, "monol_testA_E3-50_pds_rebin2")
            + HEN_FILE_EXTENSION
        )
        hen.save_as_xspec.main(command.split())
        os.path.exists(
            os.path.join(self.datadir, "monol_testA_E3-50_pds_rebin2.pha")
        )

    def test_savexspec_geom(self):
        """Test save as Xspec 2."""
        command = "{0}".format(
            os.path.join(self.datadir, "monol_test_E3-50_cpds_rebin1.03")
            + HEN_FILE_EXTENSION
        )
        hen.save_as_xspec.main(command.split())

        os.path.exists(
            os.path.join(self.datadir, "monol_test_E3-50_cpds_rebin1.03.pha")
        )
        os.path.exists(
            os.path.join(
                self.datadir, "monol_test_E3-50_cpds_rebin1.03_lags.pha"
            )
        )

    def test_plot_lin(self):
        """Test plotting with linear axes."""
        pname = (
            os.path.join(self.datadir, "monol_testA_E3-50_pds_fit")
            + HEN_FILE_EXTENSION
        )
        cname = (
            os.path.join(self.datadir, "monol_test_E3-50_cpds_fit")
            + HEN_FILE_EXTENSION
        )
        lname = (
            os.path.join(self.datadir, "monol_testA_E3-50_lc")
            + HEN_FILE_EXTENSION
        )
        hen.plot.main(
            [
                pname,
                cname,
                lname,
                "--noplot",
                "--xlin",
                "--ylin",
                "-o",
                "dummy.qdp",
            ]
        )
        hen.plot.main(
            [
                lname,
                "--noplot",
                "--axes",
                "time",
                "counts",
                "--xlin",
                "--ylin",
                "-o",
                "dummy.qdp",
            ]
        )

    def test_plot_log(self):
        """Test plotting with log axes."""
        pname = os.path.join(
            self.datadir,
            "monol_testA_E3-50_pds_rebin1.03" + HEN_FILE_EXTENSION,
        )
        cname = os.path.join(
            self.datadir,
            "monol_test_E3-50_cpds_rebin1.03" + HEN_FILE_EXTENSION,
        )
        hen.plot.main(
            [pname, cname, "--noplot", "--xlog", "--ylog", "-o", "dummy.qdp"]
        )
        hen.plot.main(
            [
                pname,
                "--noplot",
                "--axes",
                "power",
                "power_err",
                "--xlin",
                "--ylin",
                "-o",
                "dummy.qdp",
            ]
        )

    def test_plot_save_figure(self):
        """Test plotting and saving figure."""
        pname = os.path.join(
            self.datadir,
            "monol_testA_E3-50_pds_rebin1.03" + HEN_FILE_EXTENSION,
        )
        hen.plot.main(
            [
                pname,
                "--noplot",
                "--figname",
                os.path.join(
                    self.datadir, "monol_testA_E3-50_pds_rebin1.03.png"
                ),
                "-o",
                "dummy.qdp",
            ]
        )

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        def find_file_pattern_in_dir(pattern, directory):
            return glob.glob(os.path.join(directory, pattern))

        patterns = [
            "*monol_test*" + HEN_FILE_EXTENSION,
            "*lcurve*" + HEN_FILE_EXTENSION,
            "*lcurve*.txt",
            "*.log",
            "*monol_test*.dat",
            "*monol_test*.png",
            "*monol_test*.txt",
            "*monol_test_fake*.evt",
            "*bubu*",
            "*.p",
            "*.qdp",
            "*.inf",
            "*_cpds" + HEN_FILE_EXTENSION,
            "*_ev" + HEN_FILE_EXTENSION,
        ]

        file_list = []
        for pattern in patterns:
            file_list.extend(find_file_pattern_in_dir(pattern, self.datadir))

        for f in file_list:
            if os.path.exists(f):
                print("Removing " + f)
                os.remove(f)

        patterns = ["*_pds*/", "*_cpds*/", "*_sum/"]

        dir_list = []
        for pattern in patterns:
            dir_list.extend(find_file_pattern_in_dir(pattern, self.datadir))
        for f in dir_list:
            if os.path.exists(f):
                shutil.rmtree(f)
