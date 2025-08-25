import subprocess as sp
import tempfile
import numpy as np
from stingray import EventList, AveragedPowerspectrum
from hendrics.fake import main as main_fake
from hendrics.parallel import main as main_parallel


class TestParallel:
    def setup_class(cls):
        cls.tempfile = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        cls.fname = cls.tempfile.name
        print(cls.fname)

        main_fake(["-o", cls.fname, "-c", "10", "--tstart", "0", "--tstop", "10000"])
        cls.events = EventList.read(cls.fname, fmt="ogip")
        cls.pds = AveragedPowerspectrum.from_events(
            cls.events, dt=0.1, segment_size=10.0, use_common_mean=False, norm="leahy"
        )

    def test_none_version(self):
        out_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        main_parallel(
            [
                self.fname,
                "-o",
                out_file.name,
                "-b",
                "0.1",
                "-f",
                "10.0",
                "--method",
                "none",
            ]
        )
        pds = AveragedPowerspectrum.read(out_file.name)
        assert pds.m == self.pds.m
        assert np.allclose(pds.freq, self.pds.freq)
        assert np.allclose(pds.power, self.pds.power, rtol=1e-6)

    def test_multiprocessing_version(self):
        out_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        main_parallel(
            [
                self.fname,
                "-o",
                out_file.name,
                "-b",
                "0.1",
                "-f",
                "10.0",
                "--method",
                "multiprocessing",
            ]
        )
        pds = AveragedPowerspectrum.read(out_file.name)
        assert pds.m == self.pds.m
        assert np.allclose(pds.freq, self.pds.freq)
        assert np.allclose(pds.power, self.pds.power, rtol=1e-4)

    def test_mpi_version(self):
        out_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        sp.check_call(
            [
                "mpirun",
                "-n",
                "4",
                "HENparfspec",
                self.fname,
                "-o",
                out_file.name,
                "-b",
                "0.1",
                "-f",
                "10.0",
                "--method",
                "mpi",
            ]
        )
        pds = AveragedPowerspectrum.read(out_file.name)
        assert pds.m == self.pds.m
        assert np.allclose(pds.freq, self.pds.freq)
        assert np.allclose(pds.power, self.pds.power, rtol=1e-4)
