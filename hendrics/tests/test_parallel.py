import subprocess as sp
import tempfile

import numpy as np
import pytest
from stingray import AveragedPowerspectrum, EventList

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

    @pytest.mark.parametrize("method", ["none", "multiprocessing", "mpi"])
    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs"])
    def test_parallel_versions(self, method, norm):
        out_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        command = [
            self.fname,
            "-o",
            out_file.name,
            "-b",
            "0.1",
            "-f",
            "10.0",
            "--method",
            method,
            "--norm",
            norm,
        ]
        if method == "mpi":
            command = ["mpirun", "-n", "4", "HENparfspec"] + command
            sp.check_call(command)
        else:
            main_parallel(command)

        pds = AveragedPowerspectrum.read(out_file.name)
        compare_pds = self.pds.to_norm(norm) if norm != "leahy" else self.pds
        assert pds.m == compare_pds.m
        assert np.allclose(pds.freq, compare_pds.freq)
        if norm == "leahy":
            power_rtol = 1e-6
        else:
            power_rtol = 1e-2

        assert np.allclose(pds.power, compare_pds.power, rtol=power_rtol)
        assert np.allclose(pds.power_err, compare_pds.power_err, rtol=1e-2)
        assert pds.norm == compare_pds.norm
        assert np.allclose(pds.unnorm_power, compare_pds.unnorm_power, rtol=1e-2)
        assert np.isclose(pds.nphots, compare_pds.nphots, rtol=1e-2)
