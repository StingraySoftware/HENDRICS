import importlib
import os
import subprocess as sp
import tempfile

import numpy as np
import pytest
from stingray import AveragedPowerspectrum, EventList

from hendrics.fake import main as main_fake
from hendrics.parallel import main as main_parallel

HAS_MPI = importlib.util.find_spec("mpi4py") is not None
HAS_HDF5 = importlib.util.find_spec("h5py") is not None

test_cases = ["none", "multiprocessing"]
if HAS_MPI:
    from mpi4py import MPI

    test_cases.append("mpi")


@pytest.mark.skipif(not HAS_HDF5, reason="h5py is required for this test")
class TestParallel:
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        cls.fname_unsorted = os.path.join(cls.datadir, "monol_testA.evt")
        cls.fname_sorted = os.path.join(cls.datadir, "monol_testA_sort.evt")

        cls.tempfile = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        cls.fname = cls.tempfile.name
        print(cls.fname)

        main_fake(
            ["-o", cls.fname, "-c", "10", "--tstart", "0", "--tstop", "10000", "--seed", "42"]
        )
        cls.events = EventList.read(cls.fname, fmt="ogip")
        cls.pds = AveragedPowerspectrum.from_events(
            cls.events, dt=0.1, segment_size=10.0, use_common_mean=False, norm="leahy"
        )

    @pytest.mark.parametrize("method", test_cases)
    def test_parallel_versions_on_small_file(self, method):
        out_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        command = [
            self.fname_sorted,
            "-o",
            out_file.name,
            "-b",
            "0.1",
            "-f",
            "10.0",
            "--method",
            method,
            "--norm",
            "leahy",
        ]
        main_parallel(command)

    @pytest.mark.parametrize("method", test_cases)
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
        main_parallel(command)
        if method == "mpi":
            world_comm = MPI.COMM_WORLD
            my_rank = world_comm.Get_rank()
            if my_rank != 0:
                return

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

    @pytest.mark.parametrize("method", test_cases[1:])  # Skip "none" method for this test
    def test_parallel_versions_fail_unsorted(self, method, caplog):
        command = [
            self.fname_unsorted,
            "-b",
            "0.1",
            "-f",
            "10.0",
            "--method",
            method,
            "--norm",
            "leahy",
        ]
        main_parallel(command)
        if method == "mpi":
            world_comm = MPI.COMM_WORLD
            my_rank = world_comm.Get_rank()

            if my_rank != 0:
                return

        for record in caplog.records:
            if (
                record.levelname == "ERROR"
                and "The input file is probably not sorted." in record.message
            ):
                break
        else:
            raise AssertionError("Expected error message not found in logs")
