# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io.fits import Header
from stingray import (
    StingrayTimeseries,
    EventList,
    Lightcurve,
    Powerspectrum,
    Crossspectrum,
    AveragedCrossspectrum,
)


import numpy as np
import os
from hendrics.base import hen_root
from hendrics.io import load_events, save_events, save_lcurve, load_lcurve
from hendrics.io import (
    save_data,
    load_data,
    save_pds,
    load_pds,
    save_timeseries,
    load_timeseries,
)
from hendrics.io import HEN_FILE_EXTENSION, _split_high_precision_number
from hendrics.io import save_model, load_model, HAS_C256, HAS_NETCDF, HAS_H5PY
from hendrics.io import find_file_in_allowed_paths, get_file_type
from hendrics.io import save_as_ascii, save_as_qdp, read_header_key, ref_mjd
from hendrics.io import main, main_filter_events, remove_pds
from . import cleanup_test_dir
import pytest
import glob
from astropy.modeling import models
from astropy.modeling.core import Model


try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def _dummy_bad(x, z, y=0):
    return


def _dummy(x, y=0):
    return


def test_find_files_in_allowed_paths(capsys):
    with open("bu", "w") as fobj:
        print("blabla", file=fobj)
    fakepath = os.path.join("directory", "bu")
    realpath = os.path.join(".", "bu")
    foundpath = find_file_in_allowed_paths(fakepath, ["."])
    stdout, stderr = capsys.readouterr()
    assert "Parfile found at different path" in stdout
    assert foundpath == realpath
    assert find_file_in_allowed_paths("bu") == "bu"
    assert not find_file_in_allowed_paths(os.path.join("directory", "bu"))
    assert not find_file_in_allowed_paths(None)

    os.unlink("bu")


def test_high_precision_keyword():
    """Test high precision FITS keyword read."""
    from hendrics.io import high_precision_keyword_read

    hdr = {
        "MJDTESTI": 100,
        "MJDTESTF": np.longdouble(0.5),
        "CIAO": np.longdouble(0.0),
    }
    assert high_precision_keyword_read(hdr, "MJDTEST") == np.longdouble(100.5)
    assert high_precision_keyword_read(hdr, "CIAO") == np.longdouble(0.0)


class TestIO:
    """Real unit tests."""

    @classmethod
    def setup_class(cls):
        cls.dum = "bubu" + HEN_FILE_EXTENSION
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")

    def test_readfile_event(self):
        fname = os.path.join(self.datadir, "monol_testA.evt")
        main([fname])

    def test_read_header_key(self):
        fname = os.path.join(self.datadir, "monol_testA.evt")
        val = read_header_key(fname, "TSTART")
        assert val == 80000000.0

    def test_ref_mjd(self):
        fname = os.path.join(self.datadir, "monol_testA.evt")
        val = ref_mjd([fname])
        assert np.isclose(val, 55197.00076601852, atol=0.00000000001)

    def test_save_dict_data(self):
        struct = {
            "a": 0.1,
            "b": np.longdouble("123.4567890123456789"),
            "c": np.longdouble([[-0.5, 3.5]]),
            "d": 1,
        }
        save_data(struct, self.dum)
        struct2 = load_data(self.dum)
        assert np.allclose(struct["a"], struct2["a"])
        assert np.allclose(struct["b"], struct2["b"])
        assert np.allclose(struct["c"], struct2["c"])
        assert np.allclose(struct["d"], struct2["d"])

    def test_save_dict_data_bad(self):
        struct = {
            "a": 0.1,
            "b": np.longdouble("123.4567890123456789"),
            "c": np.longdouble([[-0.5, 3.5]]),
            "d": 1,
        }
        with pytest.raises(ValueError, match="Unrecognized data"):
            save_data(struct, "bubu.hdf5")

    @pytest.mark.parametrize("fmt", [".ecsv", ".hdf5"])
    def test_save_data(self, fmt):
        from astropy.table import Table

        if fmt == ".hdf5" and not HAS_H5PY:
            return

        struct = Table(
            {
                "a": [0, 0.1],
                "b": [
                    np.longdouble("123.4567890123456789"),
                    np.longdouble(0),
                ],
            }
        )
        save_data(struct, "bubu" + fmt)
        struct2 = load_data("bubu" + fmt)
        assert np.allclose(struct["a"], struct2["a"])
        assert np.allclose(struct["b"], struct2["b"])

    def test_get_file_type_astropy_format_raw(self):
        events = EventList(
            [0, 2, 3.0],
            pi=[1, 2, 3],
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        save_data(events, "bubu.ecsv")
        ftype, newdata = get_file_type("bubu.ecsv", raw_data=True)
        assert ftype == "events"
        assert isinstance(newdata, dict)

    @pytest.mark.parametrize("fmt", [HEN_FILE_EXTENSION, ".ecsv", ".hdf5"])
    def test_load_and_save_events(self, fmt, capsys):
        if fmt == ".hdf5" and not HAS_H5PY:
            return
        events = EventList(
            [0, 2, 3.0],
            pi=[1, 2, 3],
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        events.q = [2, 3, 4]
        events.cal_pi = events.pi.copy()
        events.energy = np.array([3.0, 4.0, 5.0])
        events.mission = "nustar"
        events.header = Header().tostring()
        save_events(events, "bubu" + fmt)
        events2 = load_events("bubu" + fmt)
        assert np.allclose(events.time, events2.time)
        assert np.allclose(events.cal_pi, events2.cal_pi)
        assert np.allclose(events.pi, events2.pi)
        assert np.allclose(events.q, events2.q)
        assert np.allclose(events.mjdref, events2.mjdref)
        assert np.allclose(events.gti, events2.gti)
        assert np.allclose(events.energy, events2.energy)
        assert events.header == events2.header
        assert events2.mission == events.mission
        print(events2)
        captured = capsys.readouterr()
        assert "(size 3)" in captured.out
        assert "MJD" in captured.out

    @pytest.mark.parametrize("fmt", [HEN_FILE_EXTENSION, ".ecsv", ".hdf5"])
    def test_load_and_save_timeseries(self, fmt, capsys):
        if fmt == ".hdf5" and not HAS_H5PY:
            return
        events = StingrayTimeseries(
            [0, 2, 3.0],
            pi=[1, 2, 3],
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        events.q = [2, 3, 4]
        events.cal_pi = events.pi.copy()
        events.energy = np.array([3.0, 4.0, 5.0])
        events.mission = "nustar"
        events.header = Header().tostring()
        save_timeseries(events, "bubu_ts" + fmt)
        events2 = load_timeseries("bubu_ts" + fmt)
        assert np.allclose(events.time, events2.time)
        assert np.allclose(events.cal_pi, events2.cal_pi)
        assert np.allclose(events.pi, events2.pi)
        assert np.allclose(events.q, events2.q)
        assert np.allclose(events.mjdref, events2.mjdref)
        assert np.allclose(events.gti, events2.gti)
        assert np.allclose(events.energy, events2.energy)
        assert events.header == events2.header
        assert events2.mission == events.mission
        print(events2)
        captured = capsys.readouterr()
        assert "(size 3)" in captured.out
        assert "MJD" in captured.out

    @pytest.mark.parametrize("fmt", [HEN_FILE_EXTENSION, ".ecsv", ".hdf5"])
    def test_filter_events(self, fmt):
        if fmt == ".hdf5" and not HAS_H5PY:
            return
        events = EventList(
            [0, 2, 3.0],
            pi=[1, 2, 3],
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        events.q = [2, 3, 4]
        events.cal_pi = events.pi.copy()
        events.energy = np.array([3.0, 4.0, 5.0])
        events.mission = "nustar"
        events.header = Header().tostring()
        outfile = "bubu" + fmt
        save_events(events, outfile)
        main_filter_events([outfile, "--emin", "4", "--emax", "6"])
        outfile_filt = hen_root(outfile) + f"_4-6keV" + HEN_FILE_EXTENSION
        events2 = load_events(outfile_filt)
        assert np.allclose(events.time[1:], events2.time)
        assert np.allclose(events.cal_pi[1:], events2.cal_pi)
        assert np.allclose(events.pi[1:], events2.pi)
        assert np.allclose(events.q[1:], events2.q)
        assert np.allclose(events.mjdref, events2.mjdref)
        assert np.allclose(events.gti, events2.gti)
        assert np.allclose(events.energy[1:], events2.energy)
        assert events.header == events2.header
        assert events2.mission == events.mission

    @pytest.mark.parametrize("fmt", [HEN_FILE_EXTENSION, ".ecsv", ".hdf5"])
    def test_load_and_save_lcurve(self, fmt):
        if fmt == ".hdf5" and not HAS_H5PY:
            return
        lcurve = Lightcurve(
            np.linspace(0, 10, 15),
            np.random.poisson(30, 15),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        mask = lcurve.mask
        # Monkeypatch for compatibility with old versions
        lcurve.mission = "bububu"
        lcurve.instr = "bababa"

        save_lcurve(lcurve, "bubu" + fmt)
        lcurve2 = load_lcurve("bubu" + fmt)
        assert np.allclose(lcurve.time[mask], lcurve2.time)
        assert np.allclose(lcurve.counts[mask], lcurve2.counts)
        assert np.allclose(lcurve.mjdref, lcurve2.mjdref)
        assert np.allclose(lcurve.gti, lcurve2.gti)
        assert lcurve.err_dist == lcurve2.err_dist
        assert lcurve.mission == lcurve2.mission
        assert lcurve2.mission == "bububu"
        assert lcurve.instr == lcurve2.instr

    @pytest.mark.parametrize("fmt", [HEN_FILE_EXTENSION, ".ecsv", ".hdf5"])
    def test_load_and_save_pds(self, fmt):
        if fmt == ".hdf5" and not HAS_H5PY:
            return
        pds = Powerspectrum()
        pds.freq = np.linspace(0, 10, 15)
        pds.power = np.random.poisson(30, 15)
        pds.mjdref = 54385.3254923845
        pds.gti = np.longdouble([[-0.5, 3.5]])
        pds.show_progress = True
        pds.amplitude = False

        save_pds(pds, "bubu" + fmt)
        pds2 = load_pds("bubu" + fmt)
        for attr in ["gti", "mjdref", "m", "show_progress", "amplitude"]:
            assert np.allclose(getattr(pds, attr), getattr(pds2, attr))

    @pytest.mark.parametrize("fmt", [HEN_FILE_EXTENSION, ".ecsv", ".hdf5"])
    def test_load_and_save_cpds_only(self, fmt):
        if fmt == ".hdf5" and not HAS_H5PY:
            return
        pds = Crossspectrum()
        pds.freq = np.linspace(0, 10, 15)
        pds.power = np.random.poisson(30, 15) + 1.0j
        pds.mjdref = 54385.3254923845
        pds.gti = np.longdouble([[-0.5, 3.5]])
        pds.show_progress = True
        pds.amplitude = False
        import copy

        pds.cs_all = [copy.deepcopy(pds)]
        pds.subcs = [copy.deepcopy(pds)]
        pds.pds1 = copy.deepcopy(pds)
        pds.unnorm_cs_all = [copy.deepcopy(pds)]
        pds.lc1 = Lightcurve(np.arange(2), [1, 2])

        save_pds(pds, "bubup" + fmt, no_auxil=True)
        pds2 = load_pds("bubup" + fmt)
        for attr in [
            "gti",
            "mjdref",
            "m",
            "show_progress",
            "amplitude",
            "power",
        ]:
            assert np.allclose(getattr(pds, attr), getattr(pds2, attr))

        assert not hasattr(pds2, "pds1")
        assert not hasattr(pds2, "subcs")
        assert not hasattr(pds2, "cs_all")
        assert not hasattr(pds2, "unnorm_cs_all")
        assert not hasattr(pds2, "lc1")

    @pytest.mark.parametrize("fmt", [HEN_FILE_EXTENSION, ".ecsv", ".hdf5"])
    def test_load_and_save_cpds_all(self, fmt):
        if fmt == ".hdf5" and not HAS_H5PY:
            return
        pds = Crossspectrum()
        pds.freq = np.linspace(0, 10, 15)
        pds.power = np.random.poisson(30, 15) + 1.0j
        pds.mjdref = 54385.3254923845
        pds.gti = np.longdouble([[-0.5, 3.5]])
        pds.show_progress = True
        pds.amplitude = False
        pds.cs_all = [pds]
        pds.unnorm_cs_all = [pds]
        pds.subcs = [pds]
        pds.unnorm_subcs = [pds]
        pds.lc1 = Lightcurve(np.arange(2), [1, 2])
        pds.lc2 = [Lightcurve(np.arange(2), [1, 2]), Lightcurve(np.arange(2), [3, 4])]

        with pytest.warns(
            UserWarning, match="Saving multiple light curves is not supported"
        ):
            save_pds(pds, "bubup" + fmt, save_all=True)
        pds2 = load_pds("bubup" + fmt)
        for attr in [
            "gti",
            "mjdref",
            "m",
            "show_progress",
            "amplitude",
            "power",
        ]:
            assert np.allclose(getattr(pds, attr), getattr(pds2, attr))

        assert hasattr(pds2, "cs_all")
        assert hasattr(pds2, "lc1")
        assert hasattr(pds2, "lc2")
        assert np.allclose(pds2.lc2, pds.lc2[0])
        assert hasattr(pds2, "unnorm_cs_all")
        assert not hasattr(pds2, "unnorm_subcs")
        assert not hasattr(pds2, "subcs")

    def test_load_pds_fails(self):
        pds = EventList()
        save_events(pds, self.dum)
        with pytest.raises(ValueError, match="Unrecognized data"):
            load_pds(self.dum)

    def test_load_and_save_xps(self):
        lcurve1 = Lightcurve(
            np.linspace(0, 10, 150),
            np.random.poisson(30, 150),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        lcurve2 = Lightcurve(
            np.linspace(0, 10, 150),
            np.random.poisson(30, 150),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )

        xps = AveragedCrossspectrum(lcurve1, lcurve2, 1, save_all=True)
        save_pds(xps, self.dum, save_all=True)
        xps2 = load_pds(self.dum)
        assert np.allclose(xps.gti, xps2.gti)
        assert hasattr(xps2, "cs_all")
        if hasattr(xps.cs_all[0], "power"):
            assert np.allclose(xps.cs_all[0].power, xps2.cs_all[0].power)
        else:
            assert np.allclose(xps.cs_all[0], xps2.cs_all[0])
        assert xps.m == xps2.m
        lag, lag_err = xps.time_lag()
        lag2, lag2_err = xps2.time_lag()
        assert np.allclose(lag, lag2)
        assert hasattr(xps2, "pds1")

    def test_load_and_save_xps_no_all(self):
        lcurve1 = Lightcurve(
            np.linspace(0, 10, 150),
            np.random.poisson(30, 150),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        lcurve2 = Lightcurve(
            np.linspace(0, 10, 150),
            np.random.poisson(30, 150),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )

        xps = AveragedCrossspectrum(lcurve1, lcurve2, 1)

        outfile = "small_xps" + HEN_FILE_EXTENSION

        save_pds(xps, outfile, save_all=False, no_auxil=True)
        xps2 = load_pds(outfile)
        assert np.allclose(xps.gti, xps2.gti)
        assert xps.m == xps2.m
        assert not hasattr(xps2, "pds1")
        remove_pds(outfile)

    def test_load_and_save_xps_quick(self):
        lcurve1 = Lightcurve(
            np.linspace(0, 10, 150),
            np.random.poisson(30, 150),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        lcurve2 = Lightcurve(
            np.linspace(0, 10, 150),
            np.random.poisson(30, 150),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )

        xps = AveragedCrossspectrum(lcurve1, lcurve2, 1)

        save_pds(xps, self.dum)
        xps2 = load_pds(self.dum, nosub=True)
        assert np.allclose(xps.gti, xps2.gti)
        assert xps.m == xps2.m

        assert not hasattr(xps2, "pds1")

    def test_high_precision_split1(self):
        C_I, C_F, C_l, k = _split_high_precision_number("C", np.double(0.01), 8)
        assert C_I == 1
        np.testing.assert_almost_equal(C_F, 0, 6)
        assert C_l == -2
        assert k == "double"

    def test_high_precision_split2(self):
        C_I, C_F, C_l, k = _split_high_precision_number("C", np.double(1.01), 8)
        assert C_I == 1
        np.testing.assert_almost_equal(C_F, np.double(0.01), 6)
        assert C_l == 0
        assert k == "double"

    def test_save_longcomplex(self):
        val = np.longcomplex(1.01 + 2.3j)
        data = {"val": val}
        save_data(data, "bubu" + HEN_FILE_EXTENSION)
        data_out = load_data("bubu" + HEN_FILE_EXTENSION)

        assert np.allclose(data["val"], data_out["val"])

    @pytest.mark.skipif("not HAS_C256 or not HAS_NETCDF")
    def test_save_longcomplex(self):
        val = np.complex256(1.01 + 2.3j)
        data = {"val": val}
        with pytest.warns(UserWarning, match="complex256 yet"):
            save_data(data, "bubu" + HEN_FILE_EXTENSION)
        data_out = load_data("bubu" + HEN_FILE_EXTENSION)

        assert np.allclose(data["val"], data_out["val"])

    def test_save_as_qdp(self):
        """Test saving arrays in a qdp file."""
        arrays = [np.array([0, 1, 3]), np.array([1, 4, 5])]
        errors = [np.array([1, 1, 1]), np.array([[1, 0.5], [1, 0.5], [1, 1]])]
        save_as_qdp(arrays, errors, filename=os.path.join(self.datadir, "bububu.txt"))
        save_as_qdp(
            arrays,
            errors,
            filename=os.path.join(self.datadir, "bububu.txt"),
            mode="a",
        )

    def test_save_as_ascii(self):
        """Test saving arrays in a ascii file."""
        array = np.array([0, 1, 3])
        errors = np.array([1, 1, 1])
        save_as_ascii(
            [array, errors],
            filename=os.path.join(self.datadir, "bububu.txt"),
            colnames=["array", "err"],
        )

    def test_save_as_ascii_too_many_dims(self):
        """Test saving arrays in a ascii file."""
        array = np.array([0, 1, 3])
        errors = np.array([1, 1, 1])
        dummy_out = "bububu_bad.txt"
        retval = save_as_ascii(
            np.array([[array], [errors]]),
            filename=dummy_out,
            colnames=["array", "err"],
        )
        assert not os.path.exists(dummy_out)
        assert retval == -1

    def test_save_as_ascii_simple_array(self):
        """Test saving arrays in a ascii file."""
        array = np.array([0, 1, 3])
        dummy_out = "bububu_good.txt"
        retval = save_as_ascii(array, filename=dummy_out, colnames=["array"])
        assert os.path.exists(dummy_out)
        os.unlink(dummy_out)

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(cls.datadir)
        cleanup_test_dir(".")


class TestIOModel:
    """Real unit tests."""

    @classmethod
    def setup_class(cls):
        cls.dum = "bubu" + HEN_FILE_EXTENSION
        cls.model = models.Gaussian1D() + models.Const1D(amplitude=2)

    def test_save_and_load_Astropy_model(self):
        save_model(self.model, "bubu_model.p")
        b, kind, _ = load_model("bubu_model.p")
        assert kind == "Astropy"
        assert isinstance(b, Model)
        assert np.all(self.model.parameters == b.parameters)
        assert np.all(self.model.bounds == b.bounds)
        assert np.all(self.model.fixed == b.fixed)

    def test_save_and_load_callable_model(self):
        constraints0 = {"bounds": ()}
        save_model(_dummy, "bubu_callable.p", constraints=constraints0)
        b, kind, constraints = load_model("bubu_callable.p")
        assert kind == "callable"
        assert callable(b)
        assert np.all(_dummy.__code__.co_argcount == b.__code__.co_argcount)
        assert np.all(_dummy.__defaults__ == b.__defaults__)
        assert np.all(constraints == constraints0)

    def test_save_callable_model_wrong(self):
        with pytest.raises(TypeError, match="Accepted callable models have only"):
            save_model(_dummy_bad, "callable_bad.p")
        assert not os.path.exists("callable_bad.p")

    def test_save_junk_model(self):
        a = "g"
        with pytest.raises(TypeError, match="The model has to be an Astropy model"):
            save_model(a, "bad.p", constraints={"bounds": ()})
        assert not os.path.exists("bad.p")

    def test_load_python_model_callable(self):
        modelstring = """
def model(x, a=2, b=4):
    return x * a + b

constraints = {'fixed': {'a': True}}
"""
        with open("bubu__model__.py", "w") as fobj:
            print(modelstring, file=fobj)
        b, kind, constraints = load_model("bubu__model__.py")
        assert kind == "callable"
        assert callable(b)
        assert b.__code__.co_argcount == 3
        assert b.__defaults__[0] == 2
        assert b.__defaults__[1] == 4
        assert np.all(constraints == {"fixed": {"a": True}})

    def test_load_python_model_Astropy(self):
        modelstring = """
from astropy.modeling import models
model = models.Const1D()
"""
        with open("bubu__model__2__.py", "w") as fobj:
            print(modelstring, file=fobj)
        b, kind, constraints = load_model("bubu__model__2__.py")
        assert isinstance(b, Model)
        assert kind == "Astropy"
        assert b.amplitude == 1

    def test_load_model_input_not_string(self):
        """Input is not a string"""
        with pytest.raises(
            TypeError, match="modelstring has to be an existing file name"
        ):
            b, kind, _ = load_model(1)

    def test_load_model_input_file_doesnt_exist(self):
        with pytest.raises(FileNotFoundError, match="Model file"):
            b, kind, _ = load_model("dfasjkdaslfj")

    def test_load_model_input_invalid_file_format(self):
        with open("bubu.txt", "w") as fobj:
            print(1, file=fobj)
        with pytest.raises(TypeError, match="Unknown file type"):
            b, kind, _ = load_model("bubu.txt")

    def test_load_data_fails(self):
        with pytest.raises(TypeError, match="The file type is not recognized"):
            load_data("afile.fits")

    @classmethod
    def teardown_class(cls):
        cleanup_test_dir(".")
