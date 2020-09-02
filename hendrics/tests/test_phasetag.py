import os
from astropy.io import fits
import numpy as np
import pytest
from ..phasetag import main_phasetag
from ..read_events import main as main_read
from ..io import load_events, HEN_FILE_EXTENSION, load_events_and_gtis


class TestPhasetag:
    """Real unit tests."""

    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, "data")
        cls.fits_fileA = os.path.join(cls.datadir, "monol_testA.evt")
        cls.freq = 0.1235242

    @pytest.mark.parametrize("N", [5, 6, 11, 16, 32, 41])
    def test_phase_tag(self, N):
        main_phasetag(
            [
                self.fits_fileA,
                "-f",
                str(self.freq),
                "--test",
                "--tomax",
                "-n",
                str(N),
                "--plot",
            ]
        )
        self.phasetagged = self.fits_fileA.replace(".evt", "_phasetag.evt")
        assert os.path.exists(self.phasetagged)

        # Redo to test if existing columns are preserved
        main_phasetag(
            [
                self.phasetagged,
                "-f",
                str(self.freq),
                "--test",
                "--tomax",
                "-n",
                str(N),
            ]
        )

        hdulist = fits.open(self.phasetagged)

        times = np.array(hdulist[1].data["TIME"])
        phases = np.array(hdulist[1].data["Phase"])
        orbtimes = np.array(hdulist[1].data["Orbit_bary"])
        hdulist.close()

        assert np.all(phases >= 0)
        assert np.all(phases < 1)
        # There's no orbit in these files
        assert np.allclose(times, orbtimes, rtol=1e-15)

        phase_bins = np.linspace(0, 1, N + 1)
        prof, _ = np.histogram(phases, bins=phase_bins)

        # I used --tomax
        assert np.argmax(prof) == 0
        os.unlink(self.phasetagged)

    @pytest.mark.parametrize("N", [5, 6, 11, 16, 32, 41])
    def test_phase_tag_TOA(self, N):
        main_phasetag(
            [
                self.fits_fileA,
                "-f",
                str(self.freq),
                "--test",
                "--refTOA",
                "57000.01",
                "-n",
                str(N),
                "--plot",
            ]
        )
        self.phasetagged = self.fits_fileA.replace(".evt", "_phasetag.evt")
        assert os.path.exists(self.phasetagged)

    def test_phase_tag_badexposure(self):
        with pytest.warns(UserWarning) as record:
            main_phasetag(
                [
                    self.fits_fileA,
                    "-f",
                    "0.00001",
                    "--test",
                    "--tomax",
                    "-n",
                    "1000",
                    "--plot",
                ]
            )
        assert np.any(
            [
                "Exposure has NaNs or zeros. " in r.message.args[0]
                for r in record
            ]
        )

    def test_phase_tag_invalid0(self):
        with pytest.raises(ValueError) as excinfo:
            main_phasetag([self.fits_fileA, "--test"])
        assert "Specify one between" in str(excinfo.value)

    def test_phase_tag_invalid1(self):
        with pytest.raises(ValueError) as excinfo:
            main_phasetag(
                [self.fits_fileA, "-f", "1", "--parfile", "bubu.par", "--test"]
            )
        assert "Specify only one between" in str(excinfo.value)

    def test_phase_tag_parfile(self):
        with pytest.raises(NotImplementedError) as excinfo:
            main_phasetag([self.fits_fileA, "--parfile", "bubu.par", "--test"])
        assert "This part is not yet implemented" in str(excinfo.value)
