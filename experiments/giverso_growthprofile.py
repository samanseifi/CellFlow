"""Is the colony actually transport-limited?  Measure g(r) and the front budget.

Every regime in this study has been *argued* to be diffusion-limited from two
proxies: the fitted nutrient penetration depth l, and ``active_fraction`` -- the
share of cells above the quiescence threshold. Neither is a growth rate. The
quiescence flag is binary and hysteretic, so a colony can report "13% active"
while the cells inside that band all divide at nearly the same rate as the tip,
which is the kinetics-limited regime where no front instability can exist. The
scale separation and the growth localisation are different claims, and only the
first has been measured.

This script measures the second directly.

What it computes
----------------
g(r) -- the SPECIFIC growth rate (area produced per unit colony area per unit
time), binned by radius. In CellFlow the growth law is exactly

    dA_k/dt = (a_max / 100) * (uptake_k - basal_k)      if the cell is active
              0                                          if quiescent

because a cell's area is a_max * (stored nutrient)/100. Division conserves area
(the mother halves her stored nutrient and the daughter takes the other half),
so ALL colony area production goes through this term -- there is no separate
proliferation channel to account for. g(r) is therefore the discrete model's own
Gamma / rho, the same object Giverso's continuum source is.

From g(r):

  * w        -- active-layer width, the outermost band over which g exceeds half
                its maximum. Reported as w/R and w/l.
  * interior/rim ratio -- mean g inside r < R/2 over mean g in the outer w.
                Transport-limited means this is ~0. If the interior is dividing
                at a decent fraction of the rim rate, the gradient carries no
                information and the Mullins-Sekerka feedback is dead no matter
                what the mechanics do.

Front budget (the second diagnostic)
------------------------------------
    v_budget = (integral of g dA) / (2 pi R)

is the front speed implied by the area the colony actually produces -- all of it
must leave through the perimeter. Comparing that with v_kinetic, the same
quantity evaluated as if every cell sat in the reservoir concentration, gives

    transport-limitation index  T = v_budget / v_kinetic

T << 1 means delivery sets the front speed; T ~ 1 means the cells' own kinetics
do, and the colony is in the compact-disk corner of the morphology diagram
regardless of the mechanics.

Caveat, stated because it changes how the number is read: with LINEAR uptake
(``uptake_saturation < 0``) there is no kinetic ceiling -- uptake is proportional
to n without bound -- so v_kinetic is evaluated at the reservoir value and T
measures depletion rather than saturation. With Monod on, v_kinetic is the true
V_max ceiling and T is the textbook quantity.

Run:  python experiments/giverso_growthprofile.py [regime ...]
      regimes: stage1 prolif chemo proportional starved   (default: all)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.kernels.fields import sample_field_at_cell_numba      # noqa: E402
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# The starved / Monod configuration from giverso_sectors.py, which is the only
# genuinely low-nutrient regime in the study (bc down with the quiescence
# threshold held fixed, so the rim really thins rather than the whole colony
# slowing down). It has never had its growth profile measured.
STARVED = dict(gd.PROPORTIONAL, name='starved_monod', bc_value=40.0,
               qui_threshold=10.0, uptake_km=25.0)

REGIMES = {
    'stage1': gd.STAGE1,
    'prolif': gd.PROLIFERATING,
    'chemo': gd.CHEMO,
    'proportional': gd.PROPORTIONAL,
    'starved': STARVED,
    'propfixed': gd.PROPORTIONAL_FIXED,
}


def cell_uptake_rate(cell, nutrient_field, dx):
    """Nutrient taken up per unit time, matching ``absorb_nutrient_numba``.

    Linear:  rate = consumption * n_mean * area_in_cells
    Monod :  rate = consumption * n_mean / (Km + n_mean) * area_in_cells

    The kernel integrates over the grid points inside the cell, so the mean
    field over the cell times the cell's footprint reproduces it; the shared
    footprint factor cancels in every ratio reported here, and is kept only so
    the absolute area rate is right.
    """
    n_mean = sample_field_at_cell_numba(cell.position, cell.radius,
                                        nutrient_field, dx)
    km = cell.uptake_saturation
    resp = n_mean / (km + n_mean) if km > 0 else n_mean
    return cell.consumption_rate * resp, float(n_mean)


def growth_profile(sim, regime, center, nbins=40):
    """Per-cell area production rate, binned by radius."""
    a_max = np.pi * float(sim.cells[0].max_radius) ** 2
    r, g, area, n_at = [], [], [], []
    for c in sim.cells:
        up, n_mean = cell_uptake_rate(c, sim.nutrient_field, sim.dx)
        # Quiescent cells take up nutrient but do not grow (biology kernel).
        rate = 0.0 if not c.active else (a_max / 100.0) * (up - c.basal_metabolism_rate)
        r.append(np.hypot(*(c.position - center)))
        g.append(max(rate, 0.0))
        area.append(np.pi * c.radius ** 2)
        n_at.append(n_mean)
    r, g, area, n_at = map(np.asarray, (r, g, area, n_at))

    R = float(np.percentile(r, 99))          # robust outer radius
    edges = np.linspace(0.0, R, nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    g_bin = np.zeros(nbins)
    a_bin = np.zeros(nbins)
    n_bin = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            g_bin[b] = g[m].sum()
            a_bin[b] = area[m].sum()
            n_bin[b] = n_at[m].mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        g_spec = np.where(a_bin > 0, g_bin / a_bin, np.nan)   # 1/time
    centers = 0.5 * (edges[:-1] + edges[1:])
    return dict(centers=centers, g_spec=g_spec, g_bin=g_bin, a_bin=a_bin,
                n_bin=n_bin, R=R, g_total=float(g.sum()),
                area_total=float(area.sum()), n_cells=len(r))


def layer_width(prof):
    """Outermost contiguous band where g exceeds half its maximum."""
    g, c = prof['g_spec'], prof['centers']
    ok = np.isfinite(g)
    if not ok.any() or np.nanmax(g) <= 0:
        return 0.0, np.nan
    half = 0.5 * np.nanmax(g)
    above = ok & (g >= half)
    if not above.any():
        return 0.0, np.nan
    last = np.flatnonzero(above)[-1]
    i = last
    while i > 0 and above[i - 1]:
        i -= 1
    dr = c[1] - c[0]
    return float((last - i + 1) * dr), float(c[i])


def effective_beta(sim, regime, center, R, l_meas, steps=150):
    """Place this regime on Giverso's beta axis, using their own Eq. (14).

    Their unperturbed front obeys  v* = beta * n0 * I1(R*)/I0(R*)  in units where
    length is l_c and time is 1/gamma_n. Every factor on the right is measurable
    here, so beta is not a fitting parameter -- it is read off the front speed:

        gamma_n = D / l^2            (from the fitted penetration depth)
        v_c     = l_c/t_c = D / l    (their velocity scale)
        R*      = R / l,   n0 = (nutrient at the front) / (reservoir value)
        beta    = (v_front / v_c) / (n0 * I1(R*)/I0(R*))

    This matters because beta is the knob the "starve it and it will branch"
    advice targets, and beta ~ n_c. If the measured beta is already down in the
    branching corner (their beta ~ 0.5-1) the knob is spent; if it is up near 10
    the advice has somewhere to go.
    """
    from scipy.special import ive
    r0 = float(np.percentile([np.hypot(*(c.position - center))
                              for c in sim.cells], 99))
    t0 = steps * sim.dt
    for _ in range(steps):
        sim._simulation_step()
    r1 = float(np.percentile([np.hypot(*(c.position - center))
                              for c in sim.cells], 99))
    v_front = (r1 - r0) / t0

    # Nutrient at the front: the outer 10% of the colony, relative to reservoir.
    n_front = np.mean([sample_field_at_cell_numba(c.position, c.radius,
                                                  sim.nutrient_field, sim.dx)
                       for c in sim.cells
                       if np.hypot(*(c.position - center)) > 0.9 * r1])
    n0 = n_front / regime['bc_value']

    D = regime['nutrient_D']
    v_c = D / l_meas
    Rs = R / l_meas
    # I1/I0 via the exponentially scaled forms (the exp cancels).
    bessel = float(ive(1, Rs) / ive(0, Rs))
    denom = n0 * bessel
    beta = (v_front / v_c) / denom if denom > 0 else np.nan
    return dict(v_front=float(v_front), v_c=float(v_c), n0=float(n0),
                Rs=float(Rs), beta_eff=float(beta))


def analyse(name, regime, seed=1, verbose=True):
    cfg = gd.make_config(regime, seed)
    if regime.get('uptake_km', 0) > 0:
        cfg['nutrient_uptake_saturation'] = float(regime['uptake_km'])
    sim = CellSimulation(cfg, config_name='growthprofile')
    rng = np.random.default_rng(seed)
    sim.cells = gd.seeded_colony(dict(regime, seed_eps=0.0), 1, rng)
    gd.set_phenotype(sim.cells, regime)
    if regime.get('uptake_km', 0) > 0:
        for c in sim.cells:
            c.uptake_saturation = float(regime['uptake_km'])
    n_it, converged = gd.equilibrate_nutrient(sim, regime)
    l_meas = gd.measure_length_scale(sim, regime)
    # One biology step so the quiescence flags reflect the converged field
    # (they are set inside the kernel, not by equilibration).
    sim._simulation_step()

    center = np.array([regime['L'] / 2, regime['L'] / 2])
    prof = growth_profile(sim, regime, center)
    w, w_inner = layer_width(prof)
    R = prof['R']

    g, c = prof['g_spec'], prof['centers']
    rim = np.isfinite(g) & (c >= R - max(w, c[1] - c[0]))
    core = np.isfinite(g) & (c < 0.5 * R)
    g_rim = float(np.nanmean(g[rim])) if rim.any() else np.nan
    g_core = float(np.nanmean(g[core])) if core.any() else np.nan
    ratio = g_core / g_rim if g_rim > 0 else np.nan

    # Front budget. All produced area leaves through the perimeter.
    v_budget = prof['g_total'] / (2.0 * np.pi * R)
    # Same colony, every cell at the reservoir concentration.
    a_max = np.pi * float(sim.cells[0].max_radius) ** 2
    km = float(regime.get('uptake_km', 0) or 0)
    nc = regime['bc_value']
    resp_max = nc / (km + nc) if km > 0 else nc
    g_kin = sum((a_max / 100.0) * (c_.consumption_rate * resp_max
                                   - c_.basal_metabolism_rate)
                for c_ in sim.cells)
    v_kinetic = g_kin / (2.0 * np.pi * R)
    T = v_budget / v_kinetic if v_kinetic > 0 else np.nan

    be = effective_beta(sim, regime, center, R, l_meas)

    out = dict(name=name, prof=prof, w=w, R=R, l=l_meas, ratio=ratio,
               g_rim=g_rim, g_core=g_core, v_budget=v_budget,
               v_kinetic=v_kinetic, T=T, converged=converged,
               monod=km > 0, n_cells=prof['n_cells'], **be)
    if verbose:
        print(f"  {name:<14} n={prof['n_cells']:>6}  R={R:6.1f}  l={l_meas:5.1f}  "
              f"w/R={w/R:5.3f}  w/l={w/l_meas:5.2f}  "
              f"g_core/g_rim={ratio:6.3f}  T={T:6.3f}  "
              f"R*={be['Rs']:5.1f}  n0={be['n0']:5.3f}  "
              f"beta_eff={be['beta_eff']:7.2f}"
              f"{'' if km > 0 else '   (linear uptake)'}")
        if not converged:
            print("    WARNING: nutrient field did not reach a fixed point")
    return out


def main():
    names = [a for a in sys.argv[1:] if a in REGIMES] or list(REGIMES)
    print("Transport-limitation diagnostics\n")
    print("  w/R   : active-layer width / colony radius  (thin rim => small)")
    print("  w/l   : active layer vs nutrient penetration depth")
    print("  g_core/g_rim : interior arrest (0 = fully transport-limited,")
    print("                 ~1 = kinetics-limited, no gradient information)")
    print("  T     : v_budget / v_kinetic, the front-speed flux budget\n")
    res = [analyse(n, REGIMES[n]) for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in res:
        p = r['prof']
        x = p['centers'] / r['R']
        ax[0].plot(x, p['g_spec'] / np.nanmax(p['g_spec']), '-',
                   label=f"{r['name']} (w/R={r['w']/r['R']:.2f})")
        ax[1].plot(x, p['n_bin'] / max(np.nanmax(p['n_bin']), 1e-12), '-',
                   label=r['name'])
    ax[0].axhline(0.5, color='k', lw=0.8, ls=':')
    ax[0].set(xlabel='r / R', ylabel='g(r) / max g',
              title='specific growth rate\n(dotted: the half-max level defining w)')
    ax[1].set(xlabel='r / R', ylabel='n(r) / max n',
              title='nutrient seen by the cells')
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle('Is the colony transport-limited? Growth localisation and the '
                 'nutrient profile that drives it', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = os.path.join(HERE, 'giverso_growthprofile.png')
    fig.savefig(p, dpi=115)
    print(f"\nSaved -> {p}")
    return res


if __name__ == '__main__':
    main()
