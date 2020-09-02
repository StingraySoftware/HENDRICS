# Licensed under a 3-clause BSD style license - see LICENSE.rst

from stingray.events import EventList
from stingray.lightcurve import Lightcurve
from stingray.powerspectrum import Powerspectrum, AveragedPowerspectrum
from stingray.powerspectrum import Crossspectrum, AveragedCrossspectrum
import numpy as np
import os
from hendrics.io import load_events, save_events, save_lcurve, load_lcurve
from hendrics.io import save_data, load_data, save_pds, load_pds
from hendrics.io import HEN_FILE_EXTENSION, _split_high_precision_number
from hendrics.io import save_model, load_model, HAS_C256, HAS_NETCDF
from hendrics.io import _get_additional_data, find_file_in_allowed_paths
from hendrics.io import save_as_ascii, save_as_qdp

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


def test_get_additional_data(capsys):
    from astropy.table import Table

    lctable = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert np.allclose(_get_additional_data(lctable, ["b"])["b"], [4, 5, 6])
    with pytest.warns(UserWarning) as record:
        add = _get_additional_data(lctable, ["c"])
    assert np.any(["Column c not found" in r.message.args[0] for r in record])
    assert np.allclose(add["c"], 0)


def test_get_additional_data_fits(capsys):
    from astropy.table import Table
    from astropy.io import fits

    table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
    lctable = fits.BinTableHDU(table).data
    assert np.allclose(_get_additional_data(lctable, ["b"])["b"], [4, 5, 6])
    with pytest.warns(UserWarning) as record:
        add = _get_additional_data(lctable, ["c"])
    assert np.any(["Column c not found" in r.message.args[0] for r in record])
    assert np.allclose(add["c"], 0)


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

    def test_save_data(self):
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

    def test_load_and_save_events(self):
        events = EventList(
            [0, 2, 3.0],
            pi=[1, 2, 3],
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        events.energy = np.array([3.0, 4.0, 5.0])
        events.mission = "nustar"
        save_events(events, self.dum)
        events2 = load_events(self.dum)
        assert np.allclose(events.time, events2.time)
        assert np.allclose(events.pi, events2.pi)
        assert np.allclose(events.mjdref, events2.mjdref)
        assert np.allclose(events.gti, events2.gti)
        assert np.allclose(events.energy, events2.energy)
        assert events2.mission == events.mission

    def test_load_and_save_lcurve(self):
        lcurve = Lightcurve(
            np.linspace(0, 10, 15),
            np.random.poisson(30, 15),
            mjdref=54385.3254923845,
            gti=np.longdouble([[-0.5, 3.5]]),
        )
        save_lcurve(lcurve, self.dum)
        lcurve2 = load_lcurve(self.dum)
        assert np.allclose(lcurve.time, lcurve2.time)
        assert np.allclose(lcurve.counts, lcurve2.counts)
        assert np.allclose(lcurve.mjdref, lcurve2.mjdref)
        assert np.allclose(lcurve.gti, lcurve2.gti)
        assert lcurve.err_dist == lcurve2.err_dist

    def test_load_and_save_pds(self):
        pds = Powerspectrum()
        pds.freq = np.linspace(0, 10, 15)
        pds.power = np.random.poisson(30, 15)
        pds.mjdref = 54385.3254923845
        pds.gti = np.longdouble([[-0.5, 3.5]])

        save_pds(pds, self.dum)
        pds2 = load_pds(self.dum)
        assert np.allclose(pds.gti, pds2.gti)
        assert np.allclose(pds.mjdref, pds2.mjdref)
        assert pds.m == pds2.m

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

        xps = AveragedCrossspectrum(lcurve1, lcurve2, 1)

        save_pds(xps, self.dum)
        xps2 = load_pds(self.dum)
        assert np.allclose(xps.gti, xps2.gti)
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
        save_pds(xps, outfile, save_all=False)
        xps2 = load_pds(outfile)
        assert np.allclose(xps.gti, xps2.gti)
        assert xps.m == xps2.m
        assert not hasattr(xps2, "pds1")

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
        C_I, C_F, C_l, k = _split_high_precision_number(
            "C", np.double(0.01), 8
        )
        assert C_I == 1
        np.testing.assert_almost_equal(C_F, 0, 6)
        assert C_l == -2
        assert k == "double"

    def test_high_precision_split2(self):
        C_I, C_F, C_l, k = _split_high_precision_number(
            "C", np.double(1.01), 8
        )
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
        with pytest.warns(UserWarning) as record:
            save_data(data, "bubu" + HEN_FILE_EXTENSION)
        assert "complex256 yet unsupported" in record[0].message.args[0]
        data_out = load_data("bubu" + HEN_FILE_EXTENSION)

        assert np.allclose(data["val"], data_out["val"])

    def test_save_as_qdp(self):
        """Test saving arrays in a qdp file."""
        arrays = [np.array([0, 1, 3]), np.array([1, 4, 5])]
        errors = [np.array([1, 1, 1]), np.array([[1, 0.5], [1, 0.5], [1, 1]])]
        save_as_qdp(
            arrays, errors, filename=os.path.join(self.datadir, "bububu.txt")
        )
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
        import shutil

        for dum in glob.glob("bubu*.*"):
            os.unlink(dum)
        if os.path.exists("bubu"):
            shutil.rmtree("bubu")


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
        with pytest.raises(TypeError) as record:
            save_model(_dummy_bad, "callable_bad.p")
        assert "Accepted callable models have only" in str(record.value)
        assert not os.path.exists("callable_bad.p")

    def test_save_junk_model(self):
        a = "g"
        with pytest.raises(TypeError) as record:
            save_model(a, "bad.p", constraints={"bounds": ()})
        assert "The model has to be an Astropy model or a callable" in str(
            record.value
        )
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
        with pytest.raises(TypeError) as record:
            b, kind, _ = load_model(1)
        assert "modelstring has to be an existing file name" in str(
            record.value
        )

    def test_load_model_input_file_doesnt_exist(self):
        with pytest.raises(FileNotFoundError) as record:
            b, kind, _ = load_model("dfasjkdaslfj")
        assert "Model file not found" in str(record.value)

    def test_load_model_input_invalid_file_format(self):
        with open("bubu.txt", "w") as fobj:
            print(1, file=fobj)
        with pytest.raises(TypeError) as record:
            b, kind, _ = load_model("bubu.txt")
        assert "Unknown file type" in str(record.value)

    def test_load_data_fails(self):
        with pytest.raises(TypeError) as record:
            load_data("afile.fits")
        assert "The file type is not recognized" in str(record.value)

    @classmethod
    def teardown_class(cls):
        import shutil

        for dum in glob.glob("bubu*.*"):
            os.unlink(dum)
