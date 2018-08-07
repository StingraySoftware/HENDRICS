# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function
from stingray.events import EventList
import numpy as np
import os
from hendrics.read_events import treat_event_file
from hendrics.io import HEN_FILE_EXTENSION, load_data
from hendrics.base import ref_mjd
import hendrics as hen


class TestReadEvents():
    """Real unit tests."""
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, 'data')
        cls.fits_fileA = os.path.join(cls.datadir, 'monol_testA.evt')
        cls.fits_file = os.path.join(cls.datadir, 'monol_test_fake.evt')
        hen.fake.main(['--deadtime', '1e-4', '-m', 'XMM', '-i', 'epn',
                       '--ctrate', '2000',
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
