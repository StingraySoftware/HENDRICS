from stingray.utils import jit
from math import gamma
import numpy as np
import matplotlib.pyplot as plt

from hendrics.base import r_det

@jit(nopython=True)
def sum(x):
    s = 0
    for el in x:
        s += el
    return s

@jit(nopython=True)
def factorial(n):
    return gamma(n + 1)

@jit(nopython=True)
def fn(x, n):
    return(x**(n-1) / factorial(n - 1)) * np.exp(-x)

@jit(nopython=True)
def gn(x, n):
    return sum([fn(x, l) for l in range(1, n + 1)])

@jit(nopython=True)
def Gn_back(x, n):
    return sum([gn(x, l) for l in range(1, n)])

@jit(nopython=True)
def Gn(x, n):
    return np.exp(-x) * sum([(n - l) / factorial(l) * x**l for l in range(0, n)])


@jit(nopython=True)
def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0

@jit(nopython=True)
def h(k, n, td, tb, tau):
    # Typo in Zhang+95 corrected. k * tb, not k * td 
    if k * tb < n * td:
        return 0
    return (k - n*(td + tau) / tb +
            tau / tb * Gn((k * tb - n * td)/tau, n))

INFINITE = 100
@jit(nopython=True)
def A(k, r0, td, tb, tau):
    if k == 0:
        return r0 * tb * (1 + 2 * sum([h(1, n, td, tb, tau) for n in range(1, INFINITE)]))

    eq39_sums = [h(k + 1, n, td, tb, tau) - 2 * h(k, n, td, tb, tau) + h(k - 1, n, td, tb, tau)
                 for n in range(1, INFINITE)]
    return r0 * tb * sum(eq39_sums)

def safe_A(k, r0, td, tb, tau, limit_k=60):
    if k > limit_k:
        return r0 ** 2 * tb**2
    return A(k, r0, td, tb, tau)

def check_A(rate, td, tb, max_k=100):
    """Ak ->r0**2tb**2 for k->infty"""
    tau = 1 / rate
    r0 = r_det(td, rate)

    value = r0 ** 2 * tb**2
    fig = plt.figure()
    for k in range(max_k):
        plt.scatter(k, A(k, r0, td, tb, tau), color='k')
    plt.axhline(value, ls='--', color='k')
    plt.xlabel('$k$')
    plt.ylabel('$A_k$')
    plt.savefig('check_A.png')
    plt.close(fig)


def B(k, r0, td, tb, tau):
    if k == 0:
        return 2 * (A(0, r0, td, tb, tau) - r0**2 * tb**2) / (r0*tb)

    return 4 * (A(k, r0, td, tb, tau) - r0**2 * tb**2) / (r0*tb)


def safe_B(k, r0, td, tb, tau, limit_k=60):
    if k > limit_k:
        return 0
    return B(k, r0, td, tb, tau)


def check_B(rate, td, tb, max_k=100):
    """Ak ->r0**2tb**2 for k->infty"""
    tau = 1 / rate
    r0 = r_det(td, rate)

    plt.figure()
    for k in range(max_k):
        plt.scatter(k, B(k, r0, td, tb, tau), color='k')
    plt.axhline(0, ls='--', color='k')
    plt.xlabel('$k$')
    plt.ylabel('$B_k$')


def pds_model_zhang_back(N, rate, td, tb, limit_k=60):
    tau = 1 / rate
    r0 = r_det(td, rate)
    Nph = N / tau
    P = np.zeros(N // 2)
    for j in range(N//2):
        eq8_sums = [(N - k) * safe_A(k, r0, td, tb, tau) * np.cos(2 * np.pi * j * k / N)
                    for k in range(1, N)]

        P[j] = 2 / Nph * (N * safe_A(0, r0, td, tb, tau, limit_k=limit_k) +
                          2 * sum(eq8_sums))

    maxf = 0.5 / tb
    df = maxf / len(P)
    freqs = np.arange(0, maxf, df)

    return freqs, P

def pds_model_zhang(N, rate, td, tb, limit_k=60):
    tau = 1 / rate
    r0 = r_det(td, rate)

    Nph = N / tau
    P = np.zeros(N // 2)
    for j in range(N//2):
        eq8_sums = [(N - k) / N *
                    safe_B(k, r0, td, tb, tau,
                           limit_k=limit_k) * np.cos(2 * np.pi * j * k / N)
                    for k in range(1, N)]

        P[j] = safe_B(0, r0, td, tb, tau) + sum(eq8_sums)

    maxf = 0.5 / tb
    df = maxf / len(P)
    freqs = np.arange(0, maxf, df)

    return freqs, P

def check_pds_rate(td, tb, N=100):
    """P -> 2 for rate -> 0, or tau -> infty"""

    plt.figure()
    for rate in np.logspace(2, -2, 5):
        p = pds_model_zhang_back(N, rate, td, tb)[1][1:]
        plt.scatter(1/rate, np.max(np.abs(p)), color='k')
        p = pds_model_zhang(N, rate, td, tb)[1][1:]
        plt.scatter(1/rate, np.max(np.abs(p)), color='b')

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'Max($P_j$)')
    plt.loglog()

