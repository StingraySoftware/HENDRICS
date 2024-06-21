import copy
import numpy as np
from .base import vectorize, njit, int64, float32, float64
from scipy.interpolate import interp1d
from scipy.optimize import minimize

try:
    from statsmodels.tools.numdiff import approx_hess3

    HAS_STATSM = True
except ImportError:
    HAS_STATSM = False


@vectorize([(int64,), (float32,), (float64,)])
def phases_from_zero_to_one(phase):
    """Normalize pulse phases from 0 to 1

    Examples
    --------
    >>> assert np.isclose(phases_from_zero_to_one(0.1), 0.1)
    >>> assert np.isclose(phases_from_zero_to_one(-0.9), 0.1)
    >>> assert np.isclose(phases_from_zero_to_one(0.9), 0.9)
    >>> assert np.isclose(phases_from_zero_to_one(3.1), 0.1)
    >>> assert np.allclose(phases_from_zero_to_one([0.1, 3.1, -0.9]), 0.1)
    """
    while phase > 1:
        phase -= 1.0
    while phase <= 0:
        phase += 1
    return phase


@vectorize([(int64,), (float32,), (float64,)])
def phases_around_zero(phase):
    """Normalize pulse phases from -0.5 to 0.5

    Examples
    --------
    >>> assert np.isclose(phases_around_zero(0.6), -0.4)
    >>> assert np.isclose(phases_around_zero(-0.9), 0.1)
    >>> assert np.isclose(phases_around_zero(3.9), -0.1)
    >>> assert np.allclose(phases_around_zero([0.6, -0.4]), -0.4)
    """
    ph = phase
    while ph >= 0.5:
        ph -= 1.0
    while ph < -0.5:
        ph += 1.0
    return ph


@njit()
def poisson_loglike(model, data):
    """Loglikelihood for a Poisson distribution

    Parameters
    ----------
    model : array-like
        Model
    data : array-like
        The input data
    """
    ll = 0
    for i in range(model.size):
        if model[i] <= 0:
            continue
        ll += data[i] * np.log(model[i]) - model[i]

    return -ll


def normal_loglike(model, input_data):
    """Loglikelihood for a Poisson distribution

    Parameters
    ----------
    model : array-like
        Model
    input_data : (array-like, array-like)
        Tuple containing input data and corresponding error bars
    """
    data, sigma = input_data

    return 0.5 * np.sum((model - data) ** 2 / sigma**2 + np.log(2 * np.pi * sigma**2))


def minimum_phase_diff(phase_est, phase_0):
    """Calculate a phase difference, normalized between -0.5 and 0.5."""
    diff = phase_est - phase_0
    return phases_around_zero(diff)


def get_template_func(template):
    """Get a cubic interpolation function of a pulse template.

    Parameters
    ----------
    template : array-like
        The input template profile

    Returns
    -------
    template_fun : function
        This function accepts pulse phases (even not distributed
        between 0 and 1) and returns the corresponding interpolated
        value of the pulse profile)
    """
    dph = 1 / template.size
    phases = np.linspace(0, 1, template.size + 1) + dph / 2

    allph = np.concatenate(([-dph / 2], phases))
    allt = np.concatenate((template[-1:], template, template[:1]))
    template_interp = interp1d(allph, allt, kind="cubic")

    def template_fun(x):
        ph = x - np.floor(x)
        return template_interp(ph)

    return template_fun


def normalized_template_func(template, tomax=True, subtract_min=True):
    """Get a normalized cubic interpolation function of a pulse template.

    Like in `get_template_func`, but this time the function
    returns a normalized version of the profile, having
    the maximum 1 and minimum 0.

    Parameters
    ----------
    template : array-like
        The input template profile

    Other parameters
    ----------------
    tomax: bool, default True
        Make the maximum of the profile phase 0
    subtract_min: bool, default True
        Make the minimum 0

    Returns
    -------
    template_fun : function
        The returned function
    """
    normt = copy.deepcopy(template)
    if subtract_min:
        normt -= normt.min()
    normt /= normt.max()

    template_fun = get_template_func(normt)

    dph = 1 / normt.size

    if tomax:
        start_ph = minimize(
            lambda x: -template_fun(x),
            [np.argmax(template) * dph],
            bounds=[(0, 1)],
        ).x[0]

        def new_func(x):
            return template_fun(x + start_ph)

    else:
        new_func = template_fun

    return new_func


def normalized_template(template, tomax=False, subtract_min=True):
    """Normalize a pulse template between 0 and 1.

    Parameters
    ----------
    template : array-like
        The input template profile

    Other parameters
    ----------------
    tomax: bool, default True
        Make the maximum of the profile phase 0
    subtract_min: bool, default True
        Make the minimum 0

    Examples
    --------
    >>> temp = np.sin(np.arange(0, 1, 0.01))
    >>> t2 = normalized_template(temp, tomax=True)
    >>> assert t2.max() == t2[-1]
    """
    dph = 1 / template.size
    phase = np.arange(dph / 2, 1, dph)
    return normalized_template_func(template, tomax=tomax, subtract_min=subtract_min)(
        phase
    )


# def estimate_errors(best_fit_templ, ntrial=100, profile_err=None):
#     """Estimate the error bars on the fit parameters.

#     The method used is a simple parametric bootstrap.

#     Parameters
#     ----------
#     best_fit_templ : array-like
#         The best-fit profile. This will be used to simulate
#         ntrial pulse profiles, run the fit, and estimate the
#         standard deviation of fit parameters

#     Other parameters
#     ----------------
#     ntrial: int, default 100
#         Number of simulated profiles for bootstrap
#     profile_err : float, default None
#         The error bars on each bin of the pulse profile.
#         Only relevant for Gaussian errors. If ignored,
#         the simulated profiles will be Poisson-distributed
#     """
#     bases = np.zeros(ntrial)
#     amps = np.zeros(ntrial)
#     phases = np.zeros(ntrial)

#     done = 0
#     # I interrupt after ntrial have  given a good result (done == ntrial).
#     # However, in case some of the trials fail, I give it maximum
#     # 2 * ntrial attempts.
#     for i in range(ntrial * 2):
#         roll_amount = np.random.randint(0, best_fit_templ.size)
#         dph = phases_from_zero_to_one(roll_amount / best_fit_templ.size)
#         temp = np.roll(best_fit_templ, roll_amount)

#         if profile_err is None:
#             profile = np.random.poisson(temp)
#         else:
#             profile = np.random.normal(temp, profile_err)

#         pars, _ = ml_pulsefit(profile, best_fit_templ, profile_err=profile_err)

#         diff = minimum_phase_diff(pars[2], dph)
#         if np.any(np.isnan(pars)):
#             continue
#         (bases[done], amps[done], phases[done]) = pars[0], pars[1], diff
#         done += 1
#         if done == ntrial:
#             break
#     bases = bases[:done]
#     amps = amps[:done]
#     phases = phases[:done]

#     return np.std(bases), np.std(amps), np.std(phases)


def _func_for_toa_fitting(phases, pars, template_fun):
    amp, shift = pars[:2]
    base = 0
    if len(pars) > 2:
        base = pars[2]
    return base + amp * template_fun(phases - shift)


def _guess_start_pars(profile, template, fit_base=True, mean_phase=None):
    """Guess the starting parameters for the fit.

    Examples
    --------
    >>> phases = np.linspace(0, 1, 10001)
    >>> template_fun = lambda x : 3.11 * np.exp(-(x - 0.5)**2 / (2 * 0.05**2)) + 355
    >>> template = template_fun(phases)
    >>> profile1 = 34 * template_fun(phases - 0.2) + 11.
    >>> profile2 = 53 * template_fun(phases - 0.2)
    >>> pars1 = _guess_start_pars(profile1, template, fit_base=True)
    >>> newprof1 = _func_for_toa_fitting(phases, pars1, template_fun)
    >>> assert np.allclose(profile1, newprof1, atol=1e-10)
    >>> pars2 = _guess_start_pars(profile2, template, fit_base=False)
    >>> newprof2 = _func_for_toa_fitting(phases, pars2, template_fun)
    >>> assert np.allclose(profile2, newprof2, atol=1e-10)
    """
    minp = np.min(profile)
    maxp = np.max(profile)
    mint = np.min(template)
    maxt = np.max(template)

    dph = 1 / profile.size
    if mean_phase is None:
        mean_phase = ((np.argmax(profile) - np.argmax(template))) * dph

    if fit_base:
        amp_tr = (maxp - minp) / (maxt - mint)
        x0 = (
            amp_tr,
            phases_from_zero_to_one(mean_phase),
            minp - mint * amp_tr,
        )
    else:
        x0 = (
            maxp / maxt,
            phases_from_zero_to_one(mean_phase),
        )
    return x0


def ml_pulsefit(
    profile,
    template,
    profile_err=None,
    mean_phase=None,
    fit_base=True,
    calculate_errors=False,
    ntrial=100,
):
    r"""Fit a pulse profile to a template.

    This method makes a maximum-likelihood fit of a pulse profile
    to the following function of a pulse template:
    .. math::
        f(\phi) = B + A \mathcal{T}(\phi-\phi_0)

    where :math:`\mathcal{T}` is a function that interpolates
    the input template.

    Parameters
    ----------
    profile : array-like
        The input pulse profile
    template : array-like
        The input template

    Other parameters
    ----------------
    profile_err : float, default None
        The error bars on each bin of the pulse profile.
        If specified, the maximum likelihood fitting routine
        will use a Gaussian likelihood. Otherwise, a Poisson
        likelihood will be used.
    mean_phase : float, default None
        The approximate phase of the pulse, if previously
        known. Otherwise, it is estimated from the distance
        between the maxima of the profile and the template.
    calculate_errors: bool, default False
        If True, errors are calculated with a rough bootstrap
        method.
    ntrial: int, default 100
        Number of simulated profiles for bootstrap. Only relevant
        if calculate_errors is True
    """
    template_fun = get_template_func(template)

    if profile_err is None:
        loglike = poisson_loglike
    else:
        loglike = normal_loglike

    dph = 1 / profile.size
    phases = np.arange(dph / 2, 1, dph)

    def func(pars):
        amp, shift = pars[:2]
        base = 0
        if len(pars) > 2:
            base = pars[2]
        lam = base + amp * template_fun(phases - shift)

        x = profile
        if profile_err is not None:
            x = profile, profile_err
        ll = loglike(lam, x)
        if np.isnan(ll):
            return np.inf
        return ll

    x0 = _guess_start_pars(profile, template, fit_base=fit_base, mean_phase=mean_phase)

    if fit_base:
        bounds = [(0, np.inf), (0, 1), (0, np.inf)]
    else:
        bounds = [(0, np.inf), (0, 1)]

    res = minimize(func, x0, bounds=bounds)

    if res.x[0] == 0 or np.any(np.isnan(res.x)):
        return [np.nan, np.nan, np.nan], [0, 0, 0]

    errs = [0, 0, 0]

    if calculate_errors:
        # final = res.x[0] + res.x[1] * template_fun(phases - res.x[2])
        # errs = estimate_errors(final, profile_err=profile_err, ntrial=ntrial)
        if HAS_STATSM:
            hessian_ndt = np.linalg.inv(approx_hess3(res.x, func, epsilon=dph / 10))
        else:
            hessian_ndt = res.hess_inv.todense()
        errs = np.sqrt(np.diag(hessian_ndt))

    if fit_base:
        final_pars = res.x[0], phases_around_zero(res.x[1]), res.x[2]
    else:
        final_pars = res.x[0], phases_around_zero(res.x[1]), 0
        errs = np.concatenate((errs, [0]))

    # import matplotlib.pyplot as plt

    # amp_tr, shift_tr = x0[:2]
    # base_tr = x0[2] if fit_base else 0

    # plt.figure()
    # phases_fine = np.linspace(0, 1, 300)

    # amp, shift, base = final_pars

    # shift = phases_from_zero_to_one(shift)
    # plt.title(f"{template.size} {shift}")
    # plt.plot(
    #     phases_fine, base + amp * template_fun(phases_fine - shift), label="Best fit"
    # )
    # plt.plot(
    #     phases_fine,
    #     base_tr + amp_tr * template_fun(phases_fine - shift_tr),
    #     color="grey",
    #     label="Start guess",
    # )
    # plt.plot(
    #     phases_fine,
    #     base + amp * template_fun(phases_fine),
    #     color="grey",
    #     alpha=0.5,
    #     label="Template",
    # )
    # plt.axvline(shift - errs[1])
    # plt.axvline(shift + errs[1])
    # plt.axvline(phases_from_zero_to_one(mean_phase), color="k")
    # plt.plot(phases, profile, label="Data")
    # plt.legend()
    # plt.show()
    # plt.savefig(f"{np.random.random()}.png")
    return final_pars, errs
