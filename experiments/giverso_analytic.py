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
      python experiments/giverso_analytic.py beta       # the beta / nutrient knob
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
    """Right-hand side of the dispersion equation; the root of rhs - lam is lambda(k).

    ``beta == 0`` (growth off) is short-circuited rather than multiplied through.
    At lambda = 0 the Bessel argument sqrt(lambda) vanishes and ``A_of`` is 0/0,
    so the growth term evaluates to ``0 * nan`` and poisons a result that is
    analytically just the capillary term. That case is the square-to-disc limit
    and needs to be exact, not NaN.
    """
    capill = -(sigma / Rs ** 3) * k * (k * k - 1.0)
    if beta == 0.0:
        return capill
    n0 = n0_of(Rs, Rout)
    A = A_of(lam, k, Rs, Rout, n0)
    sl1 = np.sqrt(complex(lam + 1.0))
    scale = sl1 if model == 'chemotactic' else 1.0 / sl1
    grow = beta * A * scale * _I(k + 1, sl1 * Rs)
    geom = -beta * n0 * ((1.0 + k) * _I(1, Rs) / (Rs * _I(0, Rs)) - 1.0)
    return capill + np.real(grow) + geom


def lam_of_k(k, beta, sigma, Rs, Rout, model='volumetric', allow_k1=False):
    """Solve the implicit dispersion relation for lambda at one wavenumber.

    The equation has several roots; the physical branch is the one continuously
    connected to lambda = 0 (the marginal state), so we scan outward from zero
    and take the FIRST sign change on each side. Bracketing a wide window and
    taking whatever root turns up first instead picks a spurious deep-negative
    branch -- which is what produced |lambda| ~ 2 where the paper reports 0.025.

    k = 1 is skipped by default: it is a rigid translation of the colony, not a
    shape mode, so it is excluded from "does the front finger" questions. It is
    NOT physically meaningless, though -- the paper reads it as the centre-of-mass
    asymmetry, and ``beta_sweep`` below needs it, so ``allow_k1`` re-enables it.
    """
    if k == 1 and not allow_k1:
        return np.nan

    def f(L):
        with np.errstate(over='ignore', invalid='ignore'):
            v = rhs(L, k, beta, sigma, Rs, Rout, model) - L
        return v if np.isfinite(v) else np.nan

    # POSITIVE BRANCH FIRST. For lambda > 0 the arithmetic is entirely real
    # (sqrt(lambda) real, I and K monotone) and the equation has exactly ONE
    # root -- verified over beta = 0.5..15, k = 1..40. For lambda < 0 the
    # arguments turn imaginary, I_k becomes oscillatory, and the equation grows
    # a dense set of spurious crossings (45 of them at k=1, beta=8.5) among
    # which "nearest to zero" is not meaningful. So: if the mode is unstable,
    # take the unambiguous positive root; only fall back to the scan below when
    # there is no positive root, i.e. the mode is stable and only the sign and
    # rough magnitude are being used.
    # Exactly marginal. With beta = 0 and k = 1 the capillary term vanishes too,
    # so the equation reduces to lambda = 0 identically and neither bracketing
    # scan below straddles it. That is a real neutral mode (a rigid translation
    # of a non-growing colony), not a failure, so report it as zero.
    f0 = f(0.0)
    if np.isfinite(f0) and abs(f0) < 1e-14:
        return 0.0

    pgrid = np.geomspace(1e-7, 2.0, 900)
    pv = np.array([f(x) for x in pgrid])
    pok = np.isfinite(pv)
    pg, pv = pgrid[pok], pv[pok]
    if pv.size > 1:
        sc = np.flatnonzero(np.sign(pv[:-1]) != np.sign(pv[1:]))
        if sc.size:
            try:
                return brentq(f, pg[sc[0]], pg[sc[0] + 1], xtol=1e-14)
            except (ValueError, RuntimeError):
                pass

    # Dense scan close to zero, coarsening outward. The negative window has to
    # reach past the mode's own capillary rate -- a fixed cap of -1 silently
    # returned NaN for every strongly damped mode (at sigma = 10, R* = 31 that
    # was 16 of 29 modes, since -(sigma/R*^3)k(k^2-1) reaches -9 by k = 30).
    lam_cap = (sigma / Rs ** 3) * k * (k * k - 1.0)
    span = max(1.0, 10.0 * lam_cap)
    grid = np.concatenate([
        -np.geomspace(1e-6, span, 400)[::-1], [0.0], np.geomspace(1e-6, span, 400)])
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


def curve(ks, beta, sigma, Rs, Rout, model='volumetric', allow_k1=False):
    return np.array([lam_of_k(int(k), beta, sigma, Rs, Rout, model, allow_k1)
                     for k in ks])


def front_rate(beta, Rs, Rout):
    """(dR/dt)/R for the unperturbed colony -- their Eq. (14), v* = beta n0 I1/I0.

    This is the conversion between the two amplitude conventions. The paper
    perturbs as R(theta,t) = R*(t) + eps e^(lambda t) cos(k theta), so their
    lambda is the growth rate of the ABSOLUTE lobe depth. CellFlow's dispersion
    harness reports lambda_rel, the growth rate of delta_k / R -- the shape
    deviating more and more. They differ by exactly this rate.
    """
    n0 = n0_of(Rs, Rout)
    return beta * n0 * _I(1, Rs) / (Rs * _I(0, Rs))


# ---------------------------------------------------------------------------
def beta_sweep(sigma=0.007, Rs=31.0, Rout=155.0, model='volumetric'):
    """Why does the paper say SMALL beta branches when larger beta raises lambda?

    Both statements are in the paper and both are true; they are about different
    things, and conflating them sent this study after the wrong knob.

      * Sect. 4:  "the maximum amplification rate lambda increases as beta_i
        increases (Fig. 2(b))"  -- and we reproduce that.
      * Sect. 5:  "small values of beta_i promot[e] the formation of fingers of
        decreasing thicknesses", because "for high values of beta_2, the
        characteristic wavenumber of the perturbation is k = 1".

    The reconciliation is the k = 1 mode. beta multiplies BOTH the amplification
    rate and the front velocity (Eq. 14), so raising it speeds everything up
    together and changes no shape by itself -- k_peak stays put. What it changes
    is the CONTRAST between the finger band and k = 1, the rigid translation,
    which carries no capillary penalty (the -sigma k(k^2-1) term vanishes at
    k = 1) and therefore gains the most from a larger beta. Past beta ~ 10 the
    translation outruns the whole band and the colony goes lopsided instead of
    branching.

    So "low nutrient branches" is not a statement about a stronger instability.
    It is a statement about SUPPRESSING k = 1 relative to the finger band. That
    matters here because CellFlow's colonies are k=1/k=2 dominated, which in this
    theory is the signature of large beta.
    """
    ks = np.arange(2, 41)
    betas = (0.5, 1.0, 2.0, 4.25, 8.5, 15.0)
    print(f'Giverso linear theory, {model} model, sigma={sigma}, R*={Rs}, '
          f'Rout={Rout}\n')
    print(f'{"beta":>6}{"n_c (g/l)":>11}{"(dR/dt)/R":>11}{"lam(k=1)":>10}'
          f'{"peak lam":>10}{"k_peak":>8}{"k0":>5}{"lam(1)/peak":>13}')
    print('-' * 74)
    # Their calibration (Sect. 5): beta2 = 0.5 <-> nc ~ 0.65 g/l (highly
    # branched); 4.25 <-> 5.52 g/l (dense branched/compact); 8.5 <-> ~10 g/l
    # (optimal growth, compact). beta2 = K_gamma nc / gamma_n is linear in nc, so
    # the whole column follows from the two anchors.
    nc_per_beta = 0.65 / 0.5
    rows = []
    for b in betas:
        c = curve(ks, b, sigma, Rs, Rout, model)
        l1 = lam_of_k(1, b, sigma, Rs, Rout, model, allow_k1=True)
        V = front_rate(b, Rs, Rout)
        pk = float(np.nanmax(c))
        kpk = int(ks[int(np.nanargmax(c))])
        pos = np.where(c > 0)[0]
        k0 = int(ks[pos[-1]]) if len(pos) else 0
        rows.append((b, V, l1, pk, kpk, k0, c))
        print(f'{b:>6}{b*nc_per_beta:>11.2f}{V:>11.5f}{l1:>+10.5f}{pk:>+10.5f}'
              f'{kpk:>8}{k0:>5}{l1/pk:>13.2f}')
    print('-' * 74)
    print('k_peak is flat in beta: beta sets the RATE, the geometry (R*, Rout)')
    print('sets the wavelength -- "the number of fingers is driven by Rout".')
    print('lam(1)/peak crosses 1 near beta ~ 10: past there the colony goes')
    print('lopsided (k=1) instead of fingering. THAT is what low nutrient buys.')

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    for b, V, l1, pk, kpk, k0, c in rows:
        ax[0].plot(ks, c, '-', label=f'$\\beta$={b}')
        ax[0].plot([1], [l1], 'o', ms=5, color=ax[0].lines[-1].get_color())
    ax[0].axhline(0, color='k', lw=0.8, ls=':')
    ax[0].set(xlabel='k', ylabel='$\\lambda$ (absolute amplitude)',
              title='(a) dispersion; dots at k=1 are the rigid translation')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    bb = [r[0] for r in rows]
    ax[1].plot(bb, [r[2] / r[3] for r in rows], 'ko-')
    ax[1].axhline(1.0, color='crimson', lw=1.0, ls='--')
    ax[1].set(xlabel='$\\beta$  ($\\propto$ nutrient concentration $n_c$)',
              ylabel='$\\lambda(k{=}1)\\,/\\,\\lambda(k_{peak})$',
              title='(b) translation vs finger band\n'
                    'above the dashed line the colony goes lopsided, not branched')
    ax[1].set_xscale('log'); ax[1].grid(alpha=0.3)
    for b, V, l1, pk, kpk, k0, c in rows:
        ax[1].annotate(f'$n_c\\approx${b*nc_per_beta:.1f} g/l', (b, l1 / pk),
                       textcoords='offset points', xytext=(6, -10), fontsize=7)
    fig.suptitle('Why the paper branches at SMALL $\\beta$ while its peak '
                 '$\\lambda$ grows with $\\beta$', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = os.path.join(HERE, 'giverso_analytic_beta.png')
    fig.savefig(p, dpi=115)
    print(f'\nSaved -> {p}')
    return rows


def effective_beta():
    """Read our measured sweeps back and place them on the beta axis.

    The diagnostic from ``beta_sweep`` is the ratio lambda(low k)/lambda(peak),
    which is dimensionless and so survives the fact that our time unit is not
    1/gamma_n. Our sweeps do not seed k=1 (it is a translation and the front
    analysis removes it with the centroid), so k=2 is the lowest available --
    an underestimate of the contrast, hence of the effective beta.
    """
    import glob
    print('\nmeasured sweeps, placed on the same axis:\n')
    print(f'{"sweep":<28}{"lam(k=2)":>10}{"peak":>10}{"k_peak":>8}{"ratio":>8}')
    print('-' * 64)
    for path in sorted(glob.glob(os.path.join(HERE, 'dispersion_*.json'))):
        try:
            with open(path) as fh:
                res = json.load(fh)['results']
        except (ValueError, KeyError):
            continue
        by = {}
        for r in res:
            by.setdefault(r['mode'], []).append(r)
        if not by:
            continue
        ks_m = sorted(by)
        lam = [float(np.mean([r['lambda_rel'] for r in by[k]])) for k in ks_m]
        i = int(np.argmax(lam))
        if lam[i] <= 0:
            continue                       # wholly stable sweep: no band to compare
        name = os.path.basename(path)[len('dispersion_'):-len('.json')]
        print(f'{name:<28}{lam[0]:>+10.4f}{lam[i]:>+10.4f}{ks_m[i]:>8}'
              f'{lam[0]/lam[i]:>8.2f}')
    print('-' * 64)
    print('ratio -> 1 means k=2 is as unstable as the best mode, i.e. no')
    print('separation between "lopsided" and "fingered" -- the large-beta corner.')


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
    argv = sys.argv[1:]
    if 'beta' in argv:
        beta_sweep()
        effective_beta()
    else:
        reproduce_figure2()
        if 'compare' in argv:
            compare_with_measured()
