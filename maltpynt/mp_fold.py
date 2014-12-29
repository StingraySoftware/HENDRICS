from __future__ import print_function, division, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import logging


def fold_profile_stat(profile, profile_err, meanprof=None):
    '''
    Fold profile statistics \'a la Leahy (1983)
    '''
    if meanprof is None:
        meanprof = np.mean(profile)
    return np.sum((profile - meanprof)**2 / profile_err**2)


def fold_detection_level(nbin, epsilon=0.01):
    '''
    Returns the detection level (with probability 1 - epsilon) for a folded
    profile, \'a la Leahy (1983), based on the (nbin-1) dof Chi^2 statistics
    '''
    from scipy import stats
    return stats.chi2.isf(epsilon, nbin - 1)


def dbl_cos_fit_func(p, x):
    # the frequency is fixed
    '''
    A double sinus (fundamental + 1st harmonic) used as a fit function
    '''
    startidx = 0
    base = 0
    if len(p) % 2 != 0:
        base = p[0]
        startidx = 1
    first_harm = \
        p[startidx] * np.cos(2*np.pi*x + 2*np.pi*p[startidx + 1])
    second_harm = \
        p[startidx + 2] * np.cos(4.*np.pi*x + 4*np.pi*p[startidx + 3])
    return base + first_harm + second_harm


def std_fold_fit_func(p, x):
    '''
    Chooses the fit function used in the fit
    '''

    return dbl_cos_fit_func(p, x)


def std_residuals(p, x, y):
    '''
    The residual function used in the fit
    '''
    return std_fold_fit_func(p, x) - y


def adjust_amp_phase(pars):
    '''
    Gives the phases in the interval between 0 and 1 based on the amplitude and
    phase given as input
    pars[0] is the initial amplitude
    pars[1] is the initial phase
    if amplitude is negative, it makes it positive and changes the phase
    accordingly
    '''
    if pars[0] < 0:
        pars[0] = - pars[0]
        pars[1] += 0.5
    if pars[1] < -1:
        pars[1] += np.floor(-pars[1])
    if pars[1] > 1:
        pars[1] -= np.floor(pars[1])
    pars[1] = pars[1] - np.ceil(pars[1])
    return pars


def fit_profile_with_sinusoids(profile, profile_err, debug=False, nperiods=1,
                               phaseref='default', baseline=False):
    '''
    Fits a folded profile with the std_fold_fit_func. Tries a number of
    different initial values for the fit, and returns the result of the best
    chi^2 fit
    Inputs:
        profile:     the folded profile
        profile_err: the error on the folded profile elements
        debug:       (bool, optional) print debug info
        nperiods:    (int, optional) number of periods in the folded profile.
                     Default 1.
    Output:
        fit_pars:    the best-fit parameters
        success:     whether the fit succeeded or not
        chisq:       the best chi^2
    '''
    x = np.arange(0, len(profile) * nperiods, nperiods) / float(len(profile))
    guess_pars = [max(profile) - np.mean(profile),
                  x[np.argmax(profile[:len(profile) / nperiods])] - 0.25,
                  (max(profile) - np.mean(profile)) / 2., 0.]
    startidx = 0
    if baseline:
        guess_pars = [np.mean(profile)] + guess_pars
        if debug:
            print(guess_pars)
        startidx = 1
    chisq_save = 1000000000000.
    fit_pars_save = guess_pars
    success_save = -1
    if debug:
        plt.figure('Debug profile')
        plt.errorbar(x, profile, yerr=profile_err, drawstyle='steps-mid')
        plt.plot(x, std_fold_fit_func(guess_pars, x), 'r--')

    for phase in np.arange(0., 1., 0.1):
        guess_pars[3 + startidx] = phase
        logging.debug(guess_pars)
        if debug:
            plt.plot(x, std_fold_fit_func(guess_pars, x), 'r--')
        fit_pars, success = optimize.leastsq(std_residuals, guess_pars[:],
                                             args=(x, profile))
        if debug:
            plt.plot(x, std_fold_fit_func(fit_pars, x), 'g--')
        fit_pars[startidx:startidx + 2] = \
            adjust_amp_phase(fit_pars[startidx:startidx + 2])
        fit_pars[startidx + 2:startidx + 4] = \
            adjust_amp_phase(fit_pars[startidx + 2:startidx + 4])
        chisq = np.sum((profile - std_fold_fit_func(fit_pars, x)) ** 2 /
                       profile_err ** 2) / (len(profile) - (startidx + 4))
        if debug:
            plt.plot(x, std_fold_fit_func(fit_pars, x), 'b--')
        if chisq < chisq_save:
            chisq_save = chisq
            fit_pars_save = fit_pars[:]
            success_save = success

        if debug:
            print(success_save, fit_pars_save, chisq_save)
            plt.show()
    return fit_pars_save, success_save, chisq_save


def fit_profile(profile, profile_err, debug=False, nperiods=1,
                phaseref='default', baseline=False):
    return fit_profile_with_sinusoids(profile, profile_err, debug=debug,
                                      nperiods=nperiods,
                                      phaseref=phaseref,
                                      baseline=baseline)
