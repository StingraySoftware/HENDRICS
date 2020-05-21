# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import glob

from stingray.events import EventList
import numpy as np
from hendrics.read_events import treat_event_file
from hendrics.io import HEN_FILE_EXTENSION, load_data
from hendrics.io import ref_mjd
from hendrics.fake import main
import hendrics as hen


class TestReadEvents():
    """Real unit tests."""
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, 'data')
        cls.fits_fileA = os.path.join(cls.datadir, 'monol_testA.evt')
        cls.fits_fileB = os.path.join(cls.datadir, 'monol_testA.evt')
        cls.fits_file = os.path.join(cls.datadir, 'monol_test_fake.evt')
        main(['--deadtime', '1e-4', '-m', 'XMM', '-i', 'epn',
              '--ctrate', '2000', '--mjdref', "50814.0",
              '-o', cls.fits_file])

    def test_treat_event_file_nustar(self):
        treat_event_file(self.fits_fileA)
        new_filename = 'monol_testA_nustar_fpma_ev' + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir,
                                           new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert 'instr' in data
        assert 'gti' in data
        assert 'mjdref' in data
        assert np.isclose(data['mjdref'], ref_mjd(self.fits_fileA))

    def test_treat_event_file_xmm(self):
        treat_event_file(self.fits_file)
        new_filename = 'monol_test_fake_xmm_epn_det01_ev' + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir,
                                           new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert 'instr' in data
        assert 'gti' in data
        assert 'mjdref' in data

    def test_treat_event_file_xmm_gtisplit(self):

        treat_event_file(self.fits_file, gti_split=True)
        new_filename = \
            'monol_test_fake_xmm_epn_det01_gti000_ev' + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir,
                                           new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert 'instr' in data
        assert 'gti' in data
        assert 'mjdref' in data

    def test_treat_event_file_xmm_lensplit(self):

        treat_event_file(self.fits_file, length_split=10)
        new_filename = \
            'monol_test_fake_xmm_epn_det01_chunk000_ev' + HEN_FILE_EXTENSION
        assert os.path.exists(os.path.join(self.datadir,
                                           new_filename))
        data = load_data(os.path.join(self.datadir, new_filename))
        assert 'instr' in data
        assert 'gti' in data
        assert 'mjdref' in data
        gtis = data['gti']
        lengths = np.array([g1 - g0 for (g0, g1) in gtis])
        assert np.all(lengths <= 10)

    def test_merge_events(self):
        treat_event_file(self.fits_fileA)

        filea = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_ev' + HEN_FILE_EXTENSION)
        fileb = os.path.join(
            self.datadir,
            'monol_test_fake_xmm_epn_det01_ev' + HEN_FILE_EXTENSION)

        hen.read_events.main_join([
            filea, fileb, "-o",
            os.path.join(self.datadir, "monol_merg_ev" + HEN_FILE_EXTENSION)])

        out = os.path.join(self.datadir,
                           "monol_merg_ev" + HEN_FILE_EXTENSION)
        assert os.path.exists(out)

    def test_split_events(self):
        treat_event_file(self.fits_fileA)

        filea = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_ev' + HEN_FILE_EXTENSION)

        files = hen.read_events.main_splitevents([filea, "-l", "50"])
        for f in files:
            assert os.path.exists(f)

    @classmethod
    def teardown_class(cls):
        for pattern in ["monol_*" + HEN_FILE_EXTENSION,
                        '*phasetag*', '*fake*', 'monol*.pdf']:
            files = glob.glob(
                os.path.join(cls.datadir, pattern))
            for file in files:
                os.unlink(file)

