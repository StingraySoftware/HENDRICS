import os
from astropy.io import fits
import numpy as np
import pytest
from ..phasetag import main_phasetag
from ..read_events import main as main_read
from ..io import load_events, HEN_FILE_EXTENSION, load_events_and_gtis


class TestPhasetag():
    """Real unit tests."""
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, 'data')
        cls.fits_fileA = os.path.join(cls.datadir, 'monol_testA.evt')
        cls.freq = 0.1235242

    @pytest.mark.parametrize('N', [5, 6, 11, 16])
    def test_phase_tag(self, N):
        main_phasetag([self.fits_fileA, '-f', str(self.freq), '--test',
                       '--tomax', '-n', str(N)])
        self.phasetagged = self.fits_fileA.replace('.evt', '_phasetag.evt')
        assert os.path.exists(self.phasetagged)

        # Redo to test if existing columns are preserved
        main_phasetag([self.phasetagged, '-f', str(self.freq), '--test',
                       '--tomax', '-n', str(N)])

        hdulist = fits.open(self.phasetagged)

        times = np.array(hdulist[1].data['TIME'])
        phases = np.array(hdulist[1].data['Phase'])
        orbtimes = np.array(hdulist[1].data['Orbit_bary'])
        hdulist.close()

        assert np.all(phases >= 0)
        assert np.all(phases < 1)
        # There's no orbit in these files
        assert np.allclose(times, orbtimes, rtol=1e-15)

        phase_bins = np.linspace(0, 1, N + 1)
        prof, _ = np.histogram(phases, bins=phase_bins)

        # I used --tomax
        assert np.argmax(prof) == 0
        #
        # for i, ph in enumerate(zip(phase_bins[:-1], phase_bins[1:])):
        #     good = (phases >= ph[0]) & (phases < ph[1])
        #     times_to_fold = times[good] - times[0]
        #     phases_to_fold = times_to_fold / self.freq
        #     phases_to_fold -= np.floor(phases_to_fold)
        #     prof, _ = np.histogram(phases_to_fold, bins=phase_bins)
        #     # Test that bin i has >0 events
        #     assert prof[i] > 0
        #     # Test that all the remaining bins have zero events
        #     prof[i] = 0
        #     assert np.all(prof == 0)
