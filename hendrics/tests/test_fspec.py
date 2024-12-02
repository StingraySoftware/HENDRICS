import os
import shutil

import numpy as np

try:
    import numpy.exceptions
    from numpy.exceptions import ComplexWarning
except ImportError:
    from numpy import ComplexWarning

import pytest
import stingray
from stingray import EventList

from astropy import log
from hendrics import (
    base,
    fspec,
    io,
    lcurve,
    modeling,
    plot,
    read_events,
    rebin,
    save_as_xspec,
    sum_fspec,
    timelags,
)
from hendrics.base import HENDRICS_STAR_VALUE, touch
from hendrics.fspec import calc_cpds, calc_pds
from hendrics.tests import _dummy_par

from . import cleanup_test_dir

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

HEN_FILE_EXTENSION = io.HEN_FILE_EXTENSION

log.setLevel("DEBUG")
# log.basicConfig(filename='HEN.log', level=log.DEBUG, filemode='w')


def test_pds_fails_noclobber_exists():
    touch("bububu")
    with pytest.warns(UserWarning, match="File exists, and noclobber"):
        calc_pds("bla.p", 512, outname="bububu", noclobber=True)
    os.unlink("bububu")


def test_cpds_fails_noclobber_exists():
    touch("bububu")
    with pytest.warns(UserWarning, match="File exists, and noclobber"):
        calc_cpds("bla.p", "blu.p", 512, outname="bububu", noclobber=True)
    os.unlink("bububu")


def test_distributed_pds():
    events = EventList(np.sort(np.random.uniform(0, 1000, 1000)), gti=np.asarray([[0.0, 1000]]))
    if hasattr(stingray.AveragedPowerspectrum, "from_events"):
        single_periodogram = stingray.AveragedPowerspectrum(
            events,
            segment_size=100,
            dt=0.1,
            norm="leahy",
            use_common_mean=False,
        )
    else:
        single_periodogram = stingray.AveragedPowerspectrum(
            events,
            segment_size=100,
            dt=0.1,
            norm="leahy",
        )
    pds_iterable = fspec._provide_periodograms(events, 100, 0.1, "leahy")
    pds_distr = fspec.average_periodograms(pds_iterable)
    assert np.allclose(pds_distr.power, single_periodogram.power)
    assert np.allclose(pds_distr.freq, single_periodogram.freq)
    assert pds_distr.m == single_periodogram.m


def test_distributed_cpds():
    events1 = EventList(np.sort(np.random.uniform(0, 1000, 1000)), gti=np.asarray([[0.0, 1000]]))
    events2 = EventList(np.sort(np.random.uniform(0, 1000, 1000)), gti=np.asarray([[0.0, 1000]]))
    if hasattr(stingray.AveragedCrossspectrum, "from_events"):
        single_periodogram = stingray.AveragedCrossspectrum(
            events1,
            events2,
            segment_size=100,
            dt=0.1,
            norm="leahy",
            use_common_mean=False,
        )
    else:
        single_periodogram = stingray.AveragedCrossspectrum(
            events1, events2, segment_size=100, dt=0.1, norm="leahy"
        )

    pds_iterable = fspec._provide_cross_periodograms(events1, events2, 100, 0.1, "leahy")
    pds_distr = fspec.average_periodograms(pds_iterable)
    assert np.allclose(pds_distr.power, single_periodogram.power)
    assert np.allclose(pds_distr.freq, single_periodogram.freq)
    assert pds_distr.m == single_periodogram.m


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

        cls.par = _dummy_par("bubububu.par")

        cls.ev_fileA = os.path.join(cls.datadir, "monol_testA_nustar_fpma_ev" + HEN_FILE_EXTENSION)
        cls.ev_fileB = os.path.join(cls.datadir, "monol_testB_nustar_fpmb_ev" + HEN_FILE_EXTENSION)
        for fname in [cls.ev_fileA, cls.ev_fileB]:
            if os.path.exists(fname):
                os.unlink(fname)

        cls.par = _dummy_par("bubububu.par")
        data_a, data_b = (
            os.path.join(cls.datadir, "monol_testA.evt"),
            os.path.join(cls.datadir, "monol_testB.evt"),
        )
        command = f"{data_a} {data_b}"
        read_events.main(command.split())

        command = f"{cls.ev_fileA} {cls.ev_fileB}  --nproc 2 -b -1"
        lcurve.main(command.split())
        cls.lcA = cls.ev_fileA.replace("_ev", "_lc")
        cls.lcB = cls.ev_fileB.replace("_ev", "_lc")

        pdsA = os.path.join(
            cls.datadir,
            "monol_testA_nustar_fpma_3-50keV_pds" + HEN_FILE_EXTENSION,
        )
        pdsB = os.path.join(
            cls.datadir,
            "monol_testB_nustar_fpmb_3-50keV_pds" + HEN_FILE_EXTENSION,
        )
        cpds = os.path.join(
            cls.datadir,
            "monol_test_nustar_fpm_3-50keV_cpds" + HEN_FILE_EXTENSION,
        )

        command = f"{cls.ev_fileA} {cls.ev_fileB} -f 128 -k PDS --save-all --norm leahy --emin 3 --emax 50"
        fspec.main(command.split())

        command = f"{cls.ev_fileA} {cls.ev_fileB} -f 128 -k CPDS --save-all --norm leahy --emin 3 --emax 50"
        fspec.main(command.split())

        for pds in [pdsA, pdsB, cpds]:
            assert os.path.exists(pds)

    def test_pds_leahy_emax_only(self):
        """Test PDS production."""
        evdata = self.ev_fileA

        command = f"{evdata} -f 128 -k PDS --save-all --norm leahy -b {1} --emax 50"
        fspec.main(command.split())

        out = os.path.join(
            self.datadir,
            f"monol_testA_nustar_fpma_{HENDRICS_STAR_VALUE}-50keV_pds" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(out)
        io.remove_pds(out)

    def test_pds_leahy_emin_only(self):
        """Test PDS production."""
        evdata = self.ev_fileA

        command = f"{evdata} -f 128 -k PDS --save-all --norm leahy -b {1} --emin 3"
        fspec.main(command.split())

        out = os.path.join(
            self.datadir,
            f"monol_testA_nustar_fpma_3-{HENDRICS_STAR_VALUE}keV_pds" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(out)
        io.remove_pds(out)

    def test_pds_leahy(self):
        """Test PDS production."""
        evdata = self.ev_fileA
        lcdata = self.lcA

        command = f"{evdata} -f 128. -k PDS --norm leahy -b -1"
        fspec.main(command.split())
        evout = evdata.replace("_ev", "_pds")
        assert os.path.exists(evout)
        evpds = io.load_pds(evout)
        io.remove_pds(evout)

        command = f"{lcdata} -f 128. -k PDS --save-all --norm leahy"
        fspec.main(command.split())
        lcout = lcdata.replace("_lc", "_pds")
        assert os.path.exists(lcout)
        lcpds = io.load_pds(lcout)
        io.remove_pds(lcout)

        assert np.allclose(evpds.power, lcpds.power)

    def test_pds_leahy_lombscargle(self):
        """Test PDS production."""
        evdata = self.ev_fileA
        lcdata = self.lcA

        command = f"{evdata} -k PDS --norm leahy --lombscargle -b -1"
        fspec.main(command.split())
        evout = evdata.replace("_ev", "_LS_pds")
        assert os.path.exists(evout)
        evpds = io.load_pds(evout)
        io.remove_pds(evout)

        command = f"{lcdata} -k PDS --norm leahy --lombscargle"
        fspec.main(command.split())
        lcout = lcdata.replace("_lc", "_LS_pds")
        assert os.path.exists(lcout)
        lcpds = io.load_pds(lcout)
        io.remove_pds(lcout)

        assert np.allclose(evpds.power, lcpds.power)

    def test_cpds_leahy_lombscargle(self):
        """Test PDS production."""
        evdata1 = self.ev_fileA
        evdata2 = self.ev_fileB

        command = f"{evdata1} {evdata2} -k CPDS --norm leahy --lombscargle -b -1"
        fspec.main(command.split())
        evout = evdata1.replace("fpma", "fpm").replace("testA", "test").replace("_ev", "_LS_cpds")
        assert os.path.exists(evout)
        evpds = io.load_pds(evout)
        io.remove_pds(evout)

    def test_pds_save_nothing(self):
        evdata = self.ev_fileA
        lcdata = self.lcA
        evout = evdata.replace("_ev", "_pds")
        lcout = lcdata.replace("_lc", "_pds")

        command = f"{evdata} -f 128 -k PDS --norm leahy --no-auxil -b 0.5"
        fspec.main(command.split())
        assert os.path.exists(evout)
        evpds = io.load_pds(evout)
        assert not os.path.exists(evout.replace(HEN_FILE_EXTENSION, ""))
        io.remove_pds(evout)

        command = f"{lcdata} -f 128 -k PDS --norm leahy --no-auxil "
        fspec.main(command.split())
        assert os.path.exists(lcout)
        lcpds = io.load_pds(lcout)
        assert not os.path.exists(evout.replace(HEN_FILE_EXTENSION, ""))

        io.remove_pds(lcout)

        assert np.allclose(evpds.power, lcpds.power)

    @pytest.mark.parametrize("data_kind", ["events", "lc"])
    @pytest.mark.parametrize("lombscargle", [False, True])
    def test_pds(self, data_kind, lombscargle):
        """Test PDS production."""
        if data_kind == "events":
            label = "_ev"
        else:
            label = "_lc"

        outA = os.path.join(self.datadir, "monol_testA_nustar_fpma_pds" + HEN_FILE_EXTENSION)
        outB = os.path.join(self.datadir, "monol_testB_nustar_fpmb_pds" + HEN_FILE_EXTENSION)
        if lombscargle:
            outA = outA.replace("pds", "LS_pds")
            outB = outB.replace("pds", "LS_pds")

        if os.path.exists(outA):
            io.remove_pds(outA)
        if os.path.exists(outB):
            io.remove_pds(outB)
        opts = "-f 16 --save-all --save-dyn -k PDS -b 0.5 --norm frac"
        if lombscargle:
            opts += " --lombscargle"
        data_a, data_b = (
            os.path.join(self.datadir, f"monol_testA_nustar_fpma{label}") + HEN_FILE_EXTENSION,
            os.path.join(self.datadir, f"monol_testB_nustar_fpmb{label}") + HEN_FILE_EXTENSION,
        )
        command = f"{opts} {data_a} {data_b}"
        fspec.main(command.split())

        assert os.path.exists(outA)
        assert os.path.exists(outB)

        new_pdsA = io.load_pds(outA)
        new_pdsB = io.load_pds(outB)
        for pds in [new_pdsA, new_pdsB]:
            if not lombscargle:
                assert hasattr(pds, "cs_all")
                assert len(pds.cs_all) == pds.m
            assert pds.norm == "frac"

        shutil.rmtree(outA.replace(HEN_FILE_EXTENSION, ""))
        shutil.rmtree(outB.replace(HEN_FILE_EXTENSION, ""))
        os.unlink(outA)
        os.unlink(outB)

    @pytest.mark.parametrize("data_kind", ["events", "lc"])
    def test_ignore_gti(self, data_kind):
        """Test PDS production ignoring gti."""
        if data_kind == "events":
            label = "_ev"
        else:
            label = "_lc"

        data_a, data_b = (
            os.path.join(self.datadir, f"monol_testA_nustar_fpma{label}") + HEN_FILE_EXTENSION,
            os.path.join(self.datadir, f"monol_testB_nustar_fpmb{label}") + HEN_FILE_EXTENSION,
        )
        command = f"{data_a} {data_b} -f 128 --ignore-gtis"
        fspec.main(command.split())

        outA = os.path.join(self.datadir, "monol_testA_nustar_fpma_pds" + HEN_FILE_EXTENSION)
        outB = os.path.join(self.datadir, "monol_testB_nustar_fpmb_pds" + HEN_FILE_EXTENSION)
        assert os.path.exists(outA)
        assert os.path.exists(outB)
        os.unlink(outA)
        os.unlink(outB)

    @pytest.mark.parametrize("kind", ["PDS", "CPDS"])
    def test_pds_events_big(self, kind):
        """Test PDS production."""
        labelA = "nustar_fpma_ev"
        labelB = "nustar_fpmb_ev"
        data_a, data_b = (
            os.path.join(self.datadir, f"monol_testA_{labelA}") + HEN_FILE_EXTENSION,
            os.path.join(self.datadir, f"monol_testB_{labelB}") + HEN_FILE_EXTENSION,
        )
        command = f"{data_a} {data_b} -f 16 -k {kind} --norm frac --test"
        fspec.main(command.split())

    def test_cpds_ignore_instr(self):
        """Test CPDS production."""
        out = os.path.join(self.datadir, "ignore_instr") + HEN_FILE_EXTENSION
        command = (
            f"{self.lcA} {self.lcB} -f 128 --save-dyn -k CPDS,lag --save-all --ignore-instr"
            f" -o {out} --debug"
        )

        fspec.main(command.split())

    def test_cpds_rms_norm(self):
        """Test CPDS production."""
        out = os.path.join(self.datadir, "monol_test_3-50keV_rms")
        command = f"{self.lcA} {self.lcB} -f 128 --save-dyn -k CPDS --save-all --norm rms -o {out}"

        fspec.main(command.split())

    def test_cpds_wrong_norm(self):
        """Test CPDS production."""
        out = os.path.join(self.datadir, "monol_test_3-50keV_wrong")
        command = f"{self.lcA} {self.lcB} -f 128 --save-dyn -k CPDS --norm blablabla -o {out}"
        with pytest.warns(UserWarning, match="Beware! Unknown normalization"):
            fspec.main(command.split())

    def test_cpds_dtbig(self):
        """Test CPDS production."""
        out = os.path.join(self.datadir, "monol_test_3-50keV_dtb")
        command = f"{self.lcA} {self.lcB} -f 128 --save-dyn -k CPDS --save-all --norm frac -o {out}"
        command += " -b 1"
        fspec.main(command.split())

    def test_dumpdynpds(self):
        """Test dump dynamical PDSs."""
        command = (
            "--noplot "
            + os.path.join(self.datadir, "monol_testA_3-50keV_pds_bad")
            + HEN_FILE_EXTENSION
        )
        with pytest.raises(NotImplementedError):
            fspec.dumpdyn_main(command.split())

    def test_sumpds(self):
        """Test the sum of pdss."""
        sum_fspec.main(
            [
                os.path.join(self.datadir, "monol_testA_nustar_fpma_3-50keV_pds")
                + HEN_FILE_EXTENSION,
                os.path.join(self.datadir, "monol_testB_nustar_fpmb_3-50keV_pds")
                + HEN_FILE_EXTENSION,
                "-o",
                os.path.join(self.datadir, "monol_test_sum" + HEN_FILE_EXTENSION),
            ]
        )

    def test_dumpdyncpds(self):
        """Test dump dynamical PDSs."""
        command = (
            "--noplot " + os.path.join(self.datadir, "monol_test_3-50keV_cpds") + HEN_FILE_EXTENSION
        )
        with pytest.raises(NotImplementedError):
            fspec.dumpdyn_main(command.split())

    def test_rebinpds(self):
        """Test PDS rebinning 1."""
        data = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_3-50keV_pds") + HEN_FILE_EXTENSION
        )
        command = f"{data} -r 2"
        rebin.main(command.split())
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_testA_nustar_fpma_3-50keV_pds_rebin2" + HEN_FILE_EXTENSION,
            )
        )

    def test_rebinpds_geom(self):
        """Test geometrical PDS rebinning."""
        data_a, data_b = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_3-50keV_pds") + HEN_FILE_EXTENSION,
            os.path.join(self.datadir, "monol_testB_nustar_fpmb_3-50keV_pds") + HEN_FILE_EXTENSION,
        )
        command = f"{data_a} {data_b} -r 1.03"
        rebin.main(command.split())
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_testA_nustar_fpma_3-50keV_pds_rebin1.03" + HEN_FILE_EXTENSION,
            )
        )
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_testB_nustar_fpmb_3-50keV_pds_rebin1.03" + HEN_FILE_EXTENSION,
            )
        )

    def test_rebincpds(self):
        """Test CPDS rebinning."""
        data = os.path.join(self.datadir, "monol_test_nustar_fpm_3-50keV_cpds") + HEN_FILE_EXTENSION
        command = f"{data} -r 2"
        rebin.main(command.split())
        assert os.path.exists(
            os.path.join(
                self.datadir,
                "monol_test_nustar_fpm_3-50keV_cpds_rebin2" + HEN_FILE_EXTENSION,
            )
        )

    def test_rebincpds_geom(self):
        """Test CPDS geometrical rebinning."""
        data = os.path.join(self.datadir, "monol_test_nustar_fpm_3-50keV_cpds") + HEN_FILE_EXTENSION
        command = f"{data} -r 1.03"
        rebin.main(command.split())
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_test_nustar_fpm_3-50keV_cpds_rebin1.03" + HEN_FILE_EXTENSION,
            )
        )

    def test_save_lags(self):
        fname = os.path.join(
            self.datadir,
            "monol_test_nustar_fpm_3-50keV_cpds_rebin2" + HEN_FILE_EXTENSION,
        )
        timelags.main([fname])
        out = base.hen_root(fname) + "_lags.qdp"
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
            self.datadir,
            "monol_testA_nustar_fpma_3-50keV_pds" + HEN_FILE_EXTENSION,
        )
        pdsfile2 = os.path.join(
            self.datadir,
            "monol_testB_nustar_fpmb_3-50keV_pds" + HEN_FILE_EXTENSION,
        )

        command = f"{pdsfile1} {pdsfile2} -m {modelfile} --frequency-interval 0 10"
        modeling.main_model(command.split())

        out0 = os.path.join(self.datadir, "monol_testA_nustar_fpma_3-50keV_pds_bestfit.p")
        out1 = os.path.join(self.datadir, "monol_testB_nustar_fpmb_3-50keV_pds_bestfit.p")
        assert os.path.exists(out0)
        assert os.path.exists(out1)
        m, k, c = io.load_model(
            os.path.join(self.datadir, "monol_testB_nustar_fpmb_3-50keV_pds_bestfit.p")
        )
        assert hasattr(m, "amplitude")
        os.unlink(out0)
        os.unlink(out1)

        out0 = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_3-50keV_pds_fit" + HEN_FILE_EXTENSION,
        )
        out1 = os.path.join(
            self.datadir,
            "monol_testB_nustar_fpmb_3-50keV_pds_fit" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(out0)
        assert os.path.exists(out1)
        spec = io.load_pds(out0)
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
            self.datadir,
            "monol_test_nustar_fpm_3-50keV_cpds" + HEN_FILE_EXTENSION,
        )

        command = f"{pdsfile1} -m {modelfile} --frequency-interval 0 10"

        with pytest.warns(ComplexWarning):
            modeling.main_model(command.split())

        out0 = os.path.join(self.datadir, "monol_test_nustar_fpm_3-50keV_cpds_bestfit.p")
        assert os.path.exists(out0)
        m, k, c = io.load_model(out0)
        assert hasattr(m, "amplitude")
        os.unlink(out0)

        out0 = os.path.join(
            self.datadir,
            "monol_test_nustar_fpm_3-50keV_cpds_fit" + HEN_FILE_EXTENSION,
        )
        assert os.path.exists(out0)
        spec = io.load_pds(out0)
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
            self.datadir,
            "monol_testA_nustar_fpma_3-50keV_pds" + HEN_FILE_EXTENSION,
        )
        pdsfile2 = os.path.join(
            self.datadir,
            "monol_testB_nustar_fpmb_3-50keV_pds" + HEN_FILE_EXTENSION,
        )

        command = f"{pdsfile1} {pdsfile2} -m {modelfile} --frequency-interval 0 1 9"
        with pytest.raises(ValueError, match="Invalid number of frequencies specified"):
            modeling.main_model(command.split())

    def test_savexspec(self):
        """Test save as Xspec 1."""
        data = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_3-50keV_pds_rebin2")
            + HEN_FILE_EXTENSION
        )
        command = f"{data}"
        save_as_xspec.main(command.split())
        os.path.exists(os.path.join(self.datadir, "monol_testA_nustar_fpmb_3-50keV_pds_rebin2.pha"))

    def test_savexspec_geom(self):
        """Test save as Xspec 2."""
        data = (
            os.path.join(self.datadir, "monol_test_nustar_fpm_3-50keV_cpds_rebin1.03")
            + HEN_FILE_EXTENSION
        )

        command = f"{data}"
        save_as_xspec.main(command.split())

        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_test_nustar_fpm_3-50keV_cpds_rebin1.03.pha",
            )
        )
        os.path.exists(
            os.path.join(
                self.datadir,
                "monol_test_nustar_fpm_3-50keV_cpds_rebin1.03_lags.pha",
            )
        )

    def test_plot_lin(self):
        """Test plotting with linear axes."""
        pname = (
            os.path.join(self.datadir, "monol_testA_nustar_fpma_3-50keV_pds_fit")
            + HEN_FILE_EXTENSION
        )
        cname = (
            os.path.join(self.datadir, "monol_test_nustar_fpm_3-50keV_cpds_fit")
            + HEN_FILE_EXTENSION
        )
        lname = os.path.join(self.datadir, "monol_testA_nustar_fpma_lc") + HEN_FILE_EXTENSION
        plot.main(
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
        plot.main(
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
        io.remove_pds(pname)
        io.remove_pds(cname)

    def test_plot_log(self):
        """Test plotting with log axes."""
        pname = os.path.join(
            self.datadir,
            "monol_testA_nustar_fpma_3-50keV_pds_rebin1.03" + HEN_FILE_EXTENSION,
        )
        cname = os.path.join(
            self.datadir,
            "monol_test_nustar_fpm_3-50keV_cpds_rebin1.03" + HEN_FILE_EXTENSION,
        )

        plot.main(
            [
                pname,
                cname,
                "--xlog",
                "--ylog",
                "--noplot",
                "--white-sub",
                "-o",
                "dummy.qdp",
            ]
        )
        plot.main(
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
            "monol_testA_nustar_fpma_3-50keV_pds_rebin1.03" + HEN_FILE_EXTENSION,
        )
        plot.main(
            [
                pname,
                "--noplot",
                "--figname",
                os.path.join(
                    self.datadir,
                    "monol_testA_nustar_fpma_3-50keV_pds_rebin1.03.png",
                ),
                "-o",
                "dummy.qdp",
            ]
        )

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        cleanup_test_dir(self.datadir)
        cleanup_test_dir(".")
