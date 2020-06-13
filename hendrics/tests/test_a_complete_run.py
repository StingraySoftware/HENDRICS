# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test a full run of the codes from the command line."""

import shutil
import os
import glob
import subprocess as sp

import numpy as np
from astropy import log
from astropy.tests.helper import catch_warnings
from astropy.logger import AstropyUserWarning
from astropy.tests.helper import remote_data
import pytest
from stingray.lightcurve import Lightcurve
import hendrics as hen
from hendrics.tests import _dummy_par
from hendrics.fold import HAS_PINT
from hendrics import fake, fspec, base, binary, calibrate, colors, create_gti,\
    exposure, exvar, io, lcurve, modeling, plot, read_events, rebin, \
    save_as_xspec, timelags, varenergy, sum_fspec

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

HEN_FILE_EXTENSION = hen.io.HEN_FILE_EXTENSION

log.setLevel('DEBUG')
# log.basicConfig(filename='HEN.log', level=log.DEBUG, filemode='w')


class TestFullRun(object):
    """Test how command lines work.

    Usually considered bad practice, but in this
    case I need to test the full run of the codes, and files depend on each
    other.
    Inspired by http://stackoverflow.com/questions/5387299/python-unittest-testcase-execution-order

    When command line is missing, uses some function calls
    """  # NOQA
    @classmethod
    def setup_class(cls):
        curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(curdir, 'data')
        cls.ev_fileA = os.path.join(cls.datadir,
                                            'monol_testA_nustar_fpma_ev' +
                                            HEN_FILE_EXTENSION)
        cls.par = _dummy_par("bubububu.par")

        cls.ev_fileA = os.path.join(
            cls.datadir, 'monol_testA_nustar_fpma_ev' + HEN_FILE_EXTENSION)
        cls.ev_fileB = os.path.join(
            cls.datadir, 'monol_testB_nustar_fpmb_ev' + HEN_FILE_EXTENSION)
        cls.ev_fileAcal = os.path.join(
            cls.datadir,
            'monol_testA_nustar_fpma_ev_calib' + HEN_FILE_EXTENSION)
        cls.ev_fileBcal = os.path.join(
            cls.datadir,
            'monol_testB_nustar_fpmb_ev_calib' + HEN_FILE_EXTENSION)
        cls.par = _dummy_par("bubububu.par")
        command = '{0} {1}'.format(
            os.path.join(cls.datadir, 'monol_testA.evt'),
            os.path.join(cls.datadir, 'monol_testB.evt'))
        hen.read_events.main(command.split())
        command = '{} {} -r {}'.format(
            os.path.join(cls.datadir,
                         'monol_testA_nustar_fpma_ev' + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir,
                         'monol_testB_nustar_fpmb_ev' + HEN_FILE_EXTENSION),
            os.path.join(cls.datadir, 'test.rmf'))
        hen.calibrate.main(command.split())
        cls.lcA = \
            os.path.join(os.path.join(cls.datadir,
                                      'monol_testA_E3-50_lc' +
                                      HEN_FILE_EXTENSION))
        cls.lcB = \
            os.path.join(os.path.join(cls.datadir,
                                      'monol_testB_E3-50_lc' +
                                      HEN_FILE_EXTENSION))
        command = ('{} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 '
                   '-o {}').format(cls.ev_fileAcal, cls.lcA)
        hen.lcurve.main(command.split())
        command = ('{} -e 3 50 --safe-interval 100 300  --nproc 2 -b 0.5 '
                   '-o {}').format(cls.ev_fileBcal, cls.lcB)
        hen.lcurve.main(command.split())

        cls.pdsA = os.path.join(
            cls.datadir, 'monol_testA_E3-50_pds' + HEN_FILE_EXTENSION)
        cls.pdsB = os.path.join(
            cls.datadir, 'monol_testB_E3-50_pds' + HEN_FILE_EXTENSION)
        cls.cpds = os.path.join(
            cls.datadir, 'monol_test_E3-50_cpds' + HEN_FILE_EXTENSION)

        command = \
            '{} {} -f 128 -k PDS --save-all --norm leahy'.format(
                cls.lcA, cls.lcB)
        hen.fspec.main(command.split())

        command = \
            '{} {} -f 128 -k CPDS --save-all --norm leahy'.format(
                cls.lcA, cls.lcB)
        hen.fspec.main(command.split())
        assert os.path.exists(cls.cpds)
        assert os.path.exists(cls.pdsA)
        assert os.path.exists(cls.pdsB)

    def test_scripts_are_installed(self):
        """Test only once that command line scripts are installed correctly."""
        fits_file = os.path.join(self.datadir, 'monol_testA.evt')
        command = 'HENreadfile {0}'.format(fits_file)
        sp.check_call(command.split())

    def test_save_varen_rms(self):
        fname = self.ev_fileAcal
        hen.varenergy.main([fname, "-f", "0", "100", "--energy-values",
                            "0.3", "12", "5", "lin", "--rms", "-b", "0.5",
                            "--segment-size", "128"])
        out = hen.base.hen_root(fname) + "_rms" + '.qdp'
        os.path.exists(out)

    def test_save_varen_lag(self):
        fname = self.ev_fileAcal
        hen.varenergy.main([fname, "-f", "0", "100", "--energy-values",
                            "0.3", "12", "5", "lin", "--lag", "-b", "0.5",
                            "--segment-size", "128"])
        out = hen.base.hen_root(fname) + "_lag" + '.qdp'
        os.path.exists(out)

    def test_colors_fail_uncalibrated(self):
        """Test light curve using PI filtering."""
        command = ('{0} -b 100 -e {1} {2} {2} {3}').format(
            self.ev_fileAcal, 3, 5, 10)
        with pytest.raises(ValueError) as excinfo:
            hen.colors.main(command.split())

        assert "No energy information is present " in str(excinfo.value)

    def test_colors(self):
        """Test light curve using PI filtering."""
        # calculate colors
        command = ('{0} -b 100 -e {1} {2} {2} {3}').format(
            self.ev_fileAcal, 3, 5, 10)
        hen.colors.main(command.split())

        new_filename = \
            os.path.join(self.datadir,
                         'monol_testA_nustar_fpma_E_10-5_over_5-3' +
                         HEN_FILE_EXTENSION)

        assert os.path.exists(new_filename)
        out_lc = hen.io.load_lcurve(new_filename)
        gti_to_test = hen.io.load_events(self.ev_fileA).gti
        assert np.allclose(gti_to_test, out_lc.gti)

    def test_pds_leahy_dtbig(self):
        """Test PDS production."""
        lc = self.lcA
        hen.io.main([lc])
        command = \
            '{0} -f 128 -k PDS --save-all --norm leahy -b {1}'.format(lc, 1)
        hen.fspec.main(command.split())

        assert os.path.exists(
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds' + HEN_FILE_EXTENSION))

    def test_pds_leahy(self):
        """Test PDS production."""
        lc = os.path.join(self.datadir,
                          'monol_testA_E3-50_lc') + HEN_FILE_EXTENSION
        hen.io.main([lc])
        command = \
            '{0} -f 128 -k PDS --save-all --norm leahy'.format(lc)
        hen.fspec.main(command.split())

        assert os.path.exists(
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds' + HEN_FILE_EXTENSION))

    def test_pds(self):
        """Test PDS production."""
        command = \
            '{0} {1} -f 128 --save-all --save-dyn -k PDS ' \
            '--norm frac'.format(
                os.path.join(self.datadir,
                             'monol_testA_E3-50_lc') + HEN_FILE_EXTENSION,
                os.path.join(self.datadir,
                             'monol_testB_E3-50_lc') + HEN_FILE_EXTENSION)
        hen.fspec.main(command.split())

        assert os.path.exists(
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds' + HEN_FILE_EXTENSION))
        assert os.path.exists(
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds') + HEN_FILE_EXTENSION)

    def test_pds_fits(self):
        """Test PDS production with light curves obtained from FITS files."""
        lcurve_ftools = os.path.join(self.datadir,
                                     'lcurve_ftools_lc' +
                                     HEN_FILE_EXTENSION)
        command = '{0} --save-all -f 128'.format(lcurve_ftools)
        hen.fspec.main(command.split())

    def test_pds_txt(self):
        """Test PDS production with light curves obtained from txt files."""
        lcurve_txt = os.path.join(self.datadir,
                                  'lcurve_txt_lc' +
                                  HEN_FILE_EXTENSION)
        command = '{0} --save-all -f 128'.format(lcurve_txt)
        hen.fspec.main(command.split())

    def test_cpds_rms_norm(self):
        """Test CPDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k CPDS --save-all ' \
            '--norm rms -o {2}'.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))

        hen.fspec.main(command.split())

    def test_cpds_wrong_norm(self):
        """Test CPDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k CPDS --norm blablabla -o {2}'.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))
        with pytest.warns(UserWarning) as record:
            hen.fspec.main(command.split())

        assert np.any(["Beware! Unknown normalization" in r.message.args[0]
                       for r in record])

    def test_cpds_dtbig(self):
        """Test CPDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k CPDS --save-all --norm ' \
            'frac -o {2}'.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))
        command += ' -b 1'
        hen.fspec.main(command.split())

    def test_cpds(self):
        """Test CPDS production."""
        command = \
            '{0} {1} -f 128 --save-dyn -k CPDS --save-all ' \
            '--norm frac -o {2}'.format(
                os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_testB_E3-50_lc') +
                HEN_FILE_EXTENSION,
                os.path.join(self.datadir, 'monol_test_E3-50'))
        hen.fspec.main(command.split())

    def test_dumpdynpds(self):
        """Test dump dynamical PDSs."""
        command = '--noplot ' + \
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds') + \
            HEN_FILE_EXTENSION
        with pytest.raises(NotImplementedError):
            hen.fspec.dumpdyn_main(command.split())

    def test_sumpds(self):
        """Test the sum of pdss."""
        hen.sum_fspec.main([
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds') + HEN_FILE_EXTENSION,
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds') + HEN_FILE_EXTENSION,
            '-o', os.path.join(self.datadir,
                               'monol_test_sum' + HEN_FILE_EXTENSION)])

    def test_dumpdyncpds(self):
        """Test dump dynamical PDSs."""
        command = '--noplot ' + \
            os.path.join(self.datadir,
                         'monol_test_E3-50_cpds') + \
        HEN_FILE_EXTENSION
        with pytest.raises(NotImplementedError):
            hen.fspec.dumpdyn_main(command.split())

    def test_rebinlc(self):
        """Test LC rebinning."""
        command = '{0} -r 4'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_lc') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())

    def test_rebinpds(self):
        """Test PDS rebinning 1."""
        command = '{0} -r 2'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin2' +
                                    HEN_FILE_EXTENSION))

    def test_rebinpds_geom(self):
        """Test geometrical PDS rebinning."""
        command = '{0} {1} -r 1.03'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds') +
            HEN_FILE_EXTENSION,
            os.path.join(self.datadir, 'monol_testB_E3-50_pds') +
            HEN_FILE_EXTENSION
            )
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin1.03' +
                                    HEN_FILE_EXTENSION))
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testB_E3-50_pds_rebin1.03' +
                                    HEN_FILE_EXTENSION))

    def test_rebincpds(self):
        """Test CPDS rebinning."""
        command = '{0} -r 2'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin2' +
                                    HEN_FILE_EXTENSION))

    def test_rebincpds_geom(self):
        """Test CPDS geometrical rebinning."""
        command = '{0} -r 1.03'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds') +
            HEN_FILE_EXTENSION)
        hen.rebin.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03' +
                                    HEN_FILE_EXTENSION))

    def test_save_lags(self):
        fname = os.path.join(self.datadir,
                             'monol_test_E3-50_cpds_rebin2' +
                             HEN_FILE_EXTENSION)
        hen.timelags.main([fname])
        out = hen.base.hen_root(fname) + '_lags.qdp'
        os.path.exists(out)

    def test_fit_pds(self):
        modelstring = '''
from astropy.modeling import models
model = models.Const1D()
        '''
        modelfile = 'bubu__model__.py'
        with open(modelfile, 'w') as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = \
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds' + HEN_FILE_EXTENSION)
        pdsfile2 = \
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds' + HEN_FILE_EXTENSION)

        command = '{0} {1} -m {2} --frequency-interval 0 10'.format(
            pdsfile1,
            pdsfile2,
            modelfile)
        hen.modeling.main_model(command.split())

        out0 = os.path.join(self.datadir,
                            'monol_testA_E3-50_pds_bestfit.p')
        out1 = os.path.join(self.datadir,
                            'monol_testB_E3-50_pds_bestfit.p')
        assert os.path.exists(out0)
        assert os.path.exists(out1)
        m, k, c = hen.io.load_model(
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds_bestfit.p'))
        assert hasattr(m, 'amplitude')
        os.unlink(out0)
        os.unlink(out1)

        out0 = os.path.join(self.datadir,
                            'monol_testA_E3-50_pds_fit' + HEN_FILE_EXTENSION)
        out1 = os.path.join(self.datadir,
                            'monol_testB_E3-50_pds_fit' + HEN_FILE_EXTENSION)
        assert os.path.exists(out0)
        assert os.path.exists(out1)
        spec = hen.io.load_pds(out0)
        assert hasattr(spec, 'best_fits')

    def test_fit_cpds(self):
        modelstring = '''
from astropy.modeling import models
model = models.Const1D()
        '''
        modelfile = 'bubu__model__.py'
        with open(modelfile, 'w') as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = \
            os.path.join(self.datadir,
                         'monol_test_E3-50_cpds' + HEN_FILE_EXTENSION)

        command = '{0} -m {1} --frequency-interval 0 10'.format(
            pdsfile1,
            modelfile)
        hen.modeling.main_model(command.split())

        out0 = os.path.join(self.datadir,
                            'monol_test_E3-50_cpds_bestfit.p')
        assert os.path.exists(out0)
        m, k, c = hen.io.load_model(out0)
        assert hasattr(m, 'amplitude')
        os.unlink(out0)

        out0 = \
            os.path.join(self.datadir,
                         'monol_test_E3-50_cpds_fit' + HEN_FILE_EXTENSION)
        assert os.path.exists(out0)
        spec = hen.io.load_pds(out0)
        assert hasattr(spec, 'best_fits')

    def test_fit_pds_f_no_of_intervals_invalid(self):
        modelstring = '''
from astropy.modeling import models
model = models.Const1D()
        '''
        modelfile = 'bubu__model__.py'
        with open(modelfile, 'w') as fobj:
            print(modelstring, file=fobj)
        pdsfile1 = \
            os.path.join(self.datadir,
                         'monol_testA_E3-50_pds' + HEN_FILE_EXTENSION)
        pdsfile2 = \
            os.path.join(self.datadir,
                         'monol_testB_E3-50_pds' + HEN_FILE_EXTENSION)

        command = '{0} {1} -m {2} --frequency-interval 0 1 9'.format(pdsfile1,
                                                                     pdsfile2,
                                                                     modelfile)
        with pytest.raises(ValueError) as excinfo:
            hen.modeling.main_model(command.split())

        assert "Invalid number of frequencies specified" in str(excinfo.value)

    def test_savexspec(self):
        """Test save as Xspec 1."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_testA_E3-50_pds_rebin2') +
            HEN_FILE_EXTENSION)
        hen.save_as_xspec.main(command.split())
        os.path.exists(os.path.join(self.datadir,
                                    'monol_testA_E3-50_pds_rebin2.pha'))

    def test_savexspec_geom(self):
        """Test save as Xspec 2."""
        command = '{0}'.format(
            os.path.join(self.datadir, 'monol_test_E3-50_cpds_rebin1.03') +
            HEN_FILE_EXTENSION)
        hen.save_as_xspec.main(command.split())

        os.path.exists(os.path.join(self.datadir,
                                    'monol_test_E3-50_cpds_rebin1.03.pha'))
        os.path.exists(
            os.path.join(self.datadir,
                         'monol_test_E3-50_cpds_rebin1.03_lags.pha'))

    def test_create_gti(self):
        """Test creating a GTI file."""
        fname = os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev') + \
            HEN_FILE_EXTENSION
        command = "{0} -f time>0 -c --debug".format(fname)
        hen.create_gti.main(command.split())

    def test_apply_gti(self):
        """Test applying a GTI file."""
        fname = os.path.join(self.datadir, 'monol_testA_nustar_fpma_gti') + \
            HEN_FILE_EXTENSION
        lcfname = os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev') + \
            HEN_FILE_EXTENSION
        lcoutname = os.path.join(self.datadir,
                                 'monol_testA_nustar_fpma_ev_gtifilt') + \
            HEN_FILE_EXTENSION
        command = "{0} -a {1} --debug".format(lcfname, fname)
        hen.create_gti.main(command.split())
        hen.io.load_events(lcoutname)

    def test_create_gti_and_minlen(self):
        """Test creating a GTI file and apply minimum length."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_lc_rebin4') + \
            HEN_FILE_EXTENSION
        command = "{0} -f counts>0 -c -l 10 --debug".format(fname)
        hen.create_gti.main(command.split())

    def test_create_gti_and_apply(self):
        """Test applying a GTI file and apply minimum length."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            HEN_FILE_EXTENSION
        lcfname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            HEN_FILE_EXTENSION
        command = "{0} -a {1} -l 10 --debug".format(lcfname, fname)
        hen.create_gti.main(command.split())

    def test_readfile(self):
        """Test reading and dumping a HENDRICS file."""
        fname = os.path.join(self.datadir, 'monol_testA_E3-50_rebin4_gti') + \
            HEN_FILE_EXTENSION
        command = "{0}".format(fname)

        hen.io.main(command.split())

    def test_readfile_fits(self):
        """Test reading and dumping a FITS file."""
        fitsname = os.path.join(self.datadir, 'monol_testA.evt')
        command = "{0}".format(fitsname)

        hen.io.main(command.split())

    def test_save_as_qdp(self):
        """Test saving arrays in a qdp file."""
        arrays = [np.array([0, 1, 3]), np.array([1, 4, 5])]
        errors = [np.array([1, 1, 1]), np.array([[1, 0.5], [1, 0.5], [1, 1]])]
        hen.io.save_as_qdp(arrays, errors,
                           filename=os.path.join(self.datadir,
                                                 "monol_test_qdp.txt"))
        hen.io.save_as_qdp(arrays, errors,
                           filename=os.path.join(self.datadir,
                                                 "monol_test_qdp.txt"),
                           mode='a')

    def test_save_as_ascii(self):
        """Test saving arrays in a ascii file."""
        array = np.array([0, 1, 3])
        errors = np.array([1, 1, 1])
        hen.io.save_as_ascii(
            [array, errors],
            filename=os.path.join(self.datadir, "monol_test.txt"),
            colnames=["array", "err"])

    def test_get_file_type(self):
        """Test getting file type."""
        file_list = {'events': 'monol_testA_nustar_fpma_ev',
                     'lc': 'monol_testA_E3-50_lc',
                     'pds': 'monol_testA_E3-50_pds',
                     'gti': 'monol_testA_E3-50_rebin4_gti',
                     'cpds': 'monol_test_E3-50_cpds'}
        for realtype in file_list.keys():
            fname = os.path.join(self.datadir,
                                 file_list[realtype] + HEN_FILE_EXTENSION)
            ftype, _ = hen.io.get_file_type(fname)
            assert ftype == realtype, "File types do not match"

    def test_exposure(self):
        """Test exposure calculations from unfiltered files."""
        lcname = os.path.join(self.datadir,
                              'monol_testA_E3-50_lc' + HEN_FILE_EXTENSION)
        ufname = os.path.join(self.datadir, 'monol_testA_uf.evt')
        command = "{0} {1}".format(lcname, ufname)

        hen.exposure.main(command.split())
        fname = os.path.join(self.datadir,
                             'monol_testA_E3-50_lccorr' + HEN_FILE_EXTENSION)
        assert os.path.exists(fname)
        ftype, contents = hen.io.get_file_type(fname)

        assert isinstance(contents, Lightcurve)
        assert hasattr(contents, 'expo')

    def test_plot_lin(self):
        """Test plotting with linear axes."""
        pname = os.path.join(self.datadir, 'monol_testA_E3-50_pds_fit') + \
            HEN_FILE_EXTENSION
        cname = os.path.join(self.datadir, 'monol_test_E3-50_cpds_fit') + \
            HEN_FILE_EXTENSION
        lname = os.path.join(self.datadir, 'monol_testA_E3-50_lc') + \
            HEN_FILE_EXTENSION
        hen.plot.main([pname, cname, lname, '--noplot', '--xlin', '--ylin',
                      '-o', 'dummy.qdp'])
        hen.plot.main([lname, '--noplot',
                       '--axes', 'time', 'counts', '--xlin', '--ylin',
                       '-o', 'dummy.qdp'])

    def test_plot_log(self):
        """Test plotting with log axes."""
        pname = os.path.join(
            self.datadir,
            'monol_testA_E3-50_pds_rebin1.03' + HEN_FILE_EXTENSION)
        cname = os.path.join(
            self.datadir,
            'monol_test_E3-50_cpds_rebin1.03' + HEN_FILE_EXTENSION)
        hen.plot.main([pname, cname, '--noplot', '--xlog', '--ylog',
                       '-o', 'dummy.qdp'])
        hen.plot.main([pname, '--noplot', '--axes', 'power', 'power_err',
                       '--xlin', '--ylin',
                       '-o', 'dummy.qdp'])

    def test_plot_save_figure(self):
        """Test plotting and saving figure."""
        pname = os.path.join(
            self.datadir,
            'monol_testA_E3-50_pds_rebin1.03' + HEN_FILE_EXTENSION)
        hen.plot.main([pname, '--noplot', '--figname',
                      os.path.join(self.datadir,
                                   'monol_testA_E3-50_pds_rebin1.03.png'),
                      '-o', 'dummy.qdp'])

    def test_plot_color(self):
        """Test plotting with linear axes."""
        lname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            HEN_FILE_EXTENSION
        cname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            HEN_FILE_EXTENSION
        hen.plot.main([cname, lname, '--noplot', '--xlog', '--ylog', '--CCD',
                      '-o', 'dummy.qdp'])

    def test_plot_hid(self):
        """Test plotting with linear axes."""
        # also produce a light curve with the same binning
        command = ('{0} -b 100 --energy-interval {1} {2}').format(
            os.path.join(self.datadir, 'monol_testA_nustar_fpma_ev_calib' +
                         HEN_FILE_EXTENSION), 3, 10)

        hen.lcurve.main(command.split())
        lname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E3-10_lc') + \
            HEN_FILE_EXTENSION
        os.path.exists(lname)
        cname = os.path.join(self.datadir,
                             'monol_testA_nustar_fpma_E_10-5_over_5-3') + \
            HEN_FILE_EXTENSION
        hen.plot.main([cname, lname, '--noplot', '--xlog', '--ylog', '--HID',
                      '-o', 'dummy.qdp'])

    @classmethod
    def teardown_class(self):
        """Test a full run of the scripts (command lines)."""

        def find_file_pattern_in_dir(pattern, directory):
            return glob.glob(os.path.join(directory, pattern))

        patterns = [
            '*monol_test*' + HEN_FILE_EXTENSION,
            '*lcurve*' + HEN_FILE_EXTENSION,
            '*lcurve*.txt',
            '*.log',
            '*monol_test*.dat',
            '*monol_test*.png',
            '*monol_test*.txt',
            '*monol_test_fake*.evt',
            '*bubu*',
            '*.p',
            '*.qdp',
            '*.inf'
        ]

        file_list = []
        for pattern in patterns:
            file_list.extend(
                find_file_pattern_in_dir(pattern, self.datadir)
            )

        for f in file_list:
            if os.path.exists(f):
                print("Removing " + f)
                os.remove(f)

        patterns = ['*_pds*/', '*_cpds*/', '*_sum/']

        dir_list = []
        for pattern in patterns:
            dir_list.extend(
                find_file_pattern_in_dir(pattern, self.datadir)
            )
        for f in dir_list:
            if os.path.exists(f):
                shutil.rmtree(f)
