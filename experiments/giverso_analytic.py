"""Giverso et al. (2015) analytic dispersion relation, and comparison with ours.

Implements the linear-stability result of the paper (Section 3.2 and Tables 2-4)
so the measured CellFlow dispersion can be laid against the continuum prediction
it is supposed to reproduce, rather than against a remembered summary of it.

The paper's model, in their notation
-----------------------------------
Nutrient (reaction-diffusion, LINEAR uptake, uniform in the colony):

    n_t = Dn lap(n) - gamma_n n   inside,      Dn lap(n)   outside

Colony is incompressible (drho/dt = 0) and moves by Darcy's law

    v = -Kp grad(p)                Kp = permeability = 1/(substrate friction)

with mass balance  drho/dt + rho div(v) = Gamma + div(m), giving

    chemotactic model :  Gamma = 0,          m = chi rho grad(n)
                         =>  lap(p) = -(chi/Kp) lap(n)
    volumetric model  :  Gamma = K_gamma rho n,  m = 0
                         =>  lap(p) = -(K_gamma/Kp) n

closed by Young-Laplace at the free boundary,  p = p0 - sigma_b C.

Scales: lc = sqrt(Dn/gamma_n), tc = 1/gamma_n, pc = Dn/Kp, nc = n_out(0).

Dimensionless groups (only four):
    sigma = sigma_b Kp gamma_n^(1/2) / Dn^(3/2)     surface tension (stabilising)
    beta1 = chi nc / Dn                              chemotactic model
    beta2 = K_gamma nc / gamma_n                     volumetric model
    R*    = colony radius / lc,   Rout = dish radius / lc,   q = Rout/R*

beta_i is "the ratio between the energy required for the expansion of the colony
and the energy provided by the nutrients"; larger beta raises the maximum
amplification rate. sigma stabilises: as it grows the characteristic unstable
wavenumber falls until only k = 1 survives.

Dispersion relation (their Tables 3 and 4), implicit in lambda:

  lambda = -(sigma/R*^3) k(k^2-1)
           + beta_i * A(lambda) * S_i(lambda) * I_{k+1}(sqrt(lambda+1) R*)
           - beta_i * n0 * [ (1+k) I1(R*)/(R* I0(R*)) - 1 ]

with S_1 = sqrt(lambda+1) (chemotactic) and S_2 = 1/sqrt(lambda+1) (volumetric),
and A(lambda) from their Table 2. For lambda < 0 the arguments sqrt(lambda) are
imaginary, so everything is evaluated in complex arithmetic and the (real)
root is taken.

Run:  python experiments/giverso_analytic.py            # reproduce their Fig. 2
      python experiments/giverso_analytic.py compare    # overlay our measurement
"""
import json
import os
import sys

import numpy as np
from scipy.special import ive, kve
from scipy.optimize import brentq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def _I(nu, z):
    """Modified Bessel I with the exponential scaling undone in log-space."""
    return ive(nu, z) * np.exp(np.abs(np.real(z)))


def _K(nu, z):
    return kve(nu, z) * np.exp(-z)


def n0_of(Rs, Rout):
    """Nutrient at the interface for the quasi-stationary base state (their (12))."""
    return 1.0 / (1.0 + (_I(1, Rs) / _I(0, Rs)) * Rs * np.log(Rout / Rs))


def A_of(lam, k, Rs, Rout, n0):
    """Their Table 2, general case lambda != {0, -1}."""
    sl = np.sqrt(complex(lam))
    sl1 = np.sqrt(complex(lam + 1.0))
    num = (_I(k, sl * Rs) * _K(k, sl * Rout)
           - _K(k, sl * Rs) * _I(k, sl * Rout))
    den = (sl1 * _I(k - 1, sl1 * Rs)
           * (_I(k, sl * Rout) * _K(k, sl * Rs)
              - _K(k, sl * Rout) * _I(k, sl * Rs))
           + sl * _I(k, sl1 * Rs)
           * (_I(k, sl * Rout) * _K(k - 1, sl * Rs)
              + _K(k, sl * Rout) * _I(k - 1, sl * Rs)))
    return n0 * num / den


def rhs(lam, k, beta, sigma, Rs, Rout, model):
    """Right-hand side of the dispersion equation; the root of rhs - lam is lambda(k)."""
    n0 = n0_of(Rs, Rout)
    A = A_of(lam, k, Rs, Rout, n0)
    sl1 = np.sqrt(complex(lam + 1.0))
    scale = sl1 if model == 'chemotactic' else 1.0 / sl1
    capill = -(sigma / Rs ** 3) * k * (k * k - 1.0)
    grow = beta * A * scale * _I(k + 1, sl1 * Rs)
    geom = -beta * n0 * ((1.0 + k) * _I(1, Rs) / (Rs * _I(0, Rs)) - 1.0)
    return capill + np.real(grow) + geom


def lam_of_k(k, beta, sigma, Rs, Rout, model='volumetric'):
    """Solve the implicit dispersion relation for lambda at one wavenumber.

    The equation has several roots; the physical branch is the one continuously
    connected to lambda = 0 (the marginal state), so we scan outward from zero
    and take the FIRST sign change on each side. Bracketing a wide window and
    taking whatever root turns up first instead picks a spurious deep-negative
    branch -- which is what produced |lambda| ~ 2 where the paper reports 0.025.
    """
    if k == 1:
        return np.nan                     # k=1 is a translation, not a shape mode

    def f(L):
        with np.errstate(over='ignore', invalid='ignore'):
            v = rhs(L, k, beta, sigma, Rs, Rout, model) - L
        return v if np.isfinite(v) else np.nan

    # dense scan close to zero, coarsening outward
    grid = np.concatenate([
        -np.geomspace(1e-6, 1.0, 300)[::-1], [0.0], np.geomspace(1e-6, 1.0, 300)])
    vals = np.array([f(x) for x in grid])
    ok = np.isfinite(vals)
    zero_i = int(np.argmin(np.abs(grid)))

    best = None
    for direction in (+1, -1):
        idx = range(zero_i, len(grid) - 1) if direction > 0 else range(zero_i, 0, -1)
        for i in idx:
            j = i + 1 if direction > 0 else i - 1
            if not (ok[i] and ok[j]) or np.sign(vals[i]) == np.sign(vals[j]):
                continue
            a, b = sorted((grid[i], grid[j]))
            try:
                r = brentq(f, a, b, xtol=1e-12)
            except (ValueError, RuntimeError):
                break
            if best is None or abs(r) < abs(best):
                best = r
            break
    return best if best is not None else np.nan


def curve(ks, beta, sigma, Rs, Rout, model='volumetric'):
    return np.array([lam_of_k(int(k), beta, sigma, Rs, Rout, model) for k in ks])


# ---------------------------------------------------------------------------
def reproduce_figure2():
    """Their Fig. 2: three panels varying sigma, beta and q = Rout/R*."""
    ks = np.arange(2, 31)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    for s in (0.007, 0.5, 5.0, 10.0):
        ax[0].plot(ks, curve(ks, 1.0, s, 31.0, 155.0), '-', label=f'$\\sigma$={s}')
    ax[0].set(title='(a) $\\beta=1$, $R^*=31$, $R_{out}=155$', xlabel='k',
              ylabel='$\\lambda$')

    for b in (1.0, 4.25, 8.5, 15.0):
        ax[1].plot(ks, curve(ks, b, 0.007, 31.0, 155.0), '-', label=f'$\\beta$={b}')
    ax[1].set(title='(b) $\\sigma=0.007$, $R^*=31$, $R_{out}=155$', xlabel='k')

    for q in (2.0, 5.0, 10.0, 20.0):
        ax[2].plot(ks, curve(ks, 1.0, 0.007, 31.0, 31.0 * q), '-', label=f'q={q}')
    ax[2].set(title='(c) $\\beta=1$, $\\sigma=0.007$, $R^*=31$', xlabel='k')

    for a in ax:
        a.axhline(0, color='k', lw=0.8, ls=':')
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle('Giverso et al. (2015) Fig. 2, recomputed from Tables 2-4 '
                 '(volumetric growth model)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(HERE, 'giverso_analytic_fig2.png')
    fig.savefig(p, dpi=115)
    print(f'Saved -> {p}')

    print('\nsanity checks against the published figure:')
    c = curve(ks, 1.0, 0.007, 31.0, 155.0)
    print(f'  beta=1,  sigma=0.007: lambda(k=2..5) = {np.round(c[:4], 5)}')
    print(f'    unstable band up to k = {ks[np.where(c > 0)[0][-1]] if (c>0).any() else None}'
          f' (paper: unstable at small k)')
    c15 = curve(ks, 15.0, 0.007, 31.0, 155.0)
    print(f'  beta=15, sigma=0.007: max lambda = {np.nanmax(c15):.4f} '
          f'at k = {ks[int(np.nanargmax(c15))]}   (paper panel (b) tops out ~0.08)')
    c10 = curve(ks, 1.0, 10.0, 31.0, 155.0)
    print(f'  sigma=10 (strong surface tension): max lambda = {np.nanmax(c10):.5f} '
          f'(paper: only k=1 survives, so all k>=2 should be < 0)')


def compare_with_measured():
    """Overlay our measured lambda(k) on the analytic curve at matched groups."""
    path = os.path.join(HERE, 'dispersion_proportional_chemo.json')
    if not os.path.exists(path):
        print(f'(measured sweep not found: {path})')
        return
    with open(path) as fh:
        res = json.load(fh)['results']
    by = {}
    for r in res:
        by.setdefault(r['mode'], []).append(r)
    ks_m = np.array(sorted(by))
    lam_m = np.array([np.mean([r['lambda_rel'] for r in by[k]]) for k in ks_m])

    l_diff = 8.9                       # measured diffusion length, units
    Rs = float(np.mean([r['R_start'] for r in res])) / l_diff
    Rout = (800.0 / 2.0) / l_diff      # box half-width in units of lc
    print(f'\nmatched geometry: R* = {Rs:.1f}, Rout = {Rout:.1f}, '
          f'q = {Rout/Rs:.1f}   (paper Fig.2 used R*=31, q=5)')

    ks = np.arange(2, 21)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ks_m, lam_m, 'ko-', label='CellFlow measured (proportional)')
    for b, s in ((1.0, 0.007), (4.0, 0.05), (8.0, 0.2)):
        ax.plot(ks, curve(ks, b, s, Rs, Rout, 'chemotactic'), '--',
                label=f'Giverso chemotactic $\\beta$={b}, $\\sigma$={s}')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set(xlabel='mode k', ylabel='$\\lambda$ (own units)',
           title='Measured vs analytic dispersion, matched $R^*$ and $q$\n'
                 '(vertical scales differ: our time unit is not $1/\\gamma_n$)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(HERE, 'giverso_analytic_compare.png')
    fig.savefig(p, dpi=115)
    print(f'Saved -> {p}')


if __name__ == '__main__':
    reproduce_figure2()
    if 'compare' in sys.argv[1:]:
        compare_with_measured()
