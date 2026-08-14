"""Does the front advance faster where the nutrient flux is higher?

This is the single measurement that decides what is missing. Mullins-Sekerka
instability needs the front's normal velocity to respond to the local flux:

    V(theta)  proportional to  flux(theta),

so that a protrusion, which intercepts more nutrient, outruns the valleys and the
perturbation amplifies. The measured dispersion relations show lambda(k) LINEAR in
k (r^2 ~ 0.995) rather than the Mullins-Sekerka k - k^3 shape, which says the
destabilising +V*k term is absent altogether. This script tests that directly.

Method: seed a lobed front with amplitude LARGER than the nutrient penetration
depth l, so the tips genuinely stick out of the diffusive boundary layer and flux
focusing has every chance to act. Then, per angular bin, measure

    flux proxy : nutrient concentration and |grad c| just ahead of the front
    response   : local front advance velocity V = dR/dt

and report the ELASTICITY  E = (dV/V) / (dflux/flux)  -- the relative advance
response to a relative flux variation. Mullins-Sekerka needs E ~ 1. E ~ 0 means
the front advances at the same speed regardless of how much nutrient it is
getting, i.e. there is no flux focusing to destabilise anything.

Run:  python experiments/giverso_fluxresponse.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.analysis.front import main_cluster_mask, front_radii  # noqa: E402
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

N_BINS = 120
STEPS = 120
MODE = 10


def sample_ahead(sim, center, R, ahead):
    """Nutrient c and |grad c| sampled just outside the front in each bin."""
    g = sim.grid_resolution
    gy, gx = np.gradient(sim.nutrient_field, sim.dx)
    th = np.linspace(-np.pi, np.pi, len(R), endpoint=False) + np.pi / len(R)
    x = center[0] + (R + ahead) * np.cos(th)
    y = center[1] + (R + ahead) * np.sin(th)
    i = np.clip((x / sim.dx).astype(int), 0, g - 1)
    j = np.clip((y / sim.dx).astype(int), 0, g - 1)
    c = sim.nutrient_field[j, i]
    grad = np.hypot(gx[j, i], gy[j, i])
    return c, grad


def elasticity(flux, V):
    """(dV/V) / (dflux/flux) from a linear fit, plus Pearson r."""
    ok = np.isfinite(flux) & np.isfinite(V) & (flux > 0)
    f, v = flux[ok], V[ok]
    if f.size < 10 or f.std() == 0:
        return np.nan, np.nan
    slope = np.polyfit(f, v, 1)[0]
    E = slope * f.mean() / v.mean() if v.mean() != 0 else np.nan
    r = np.corrcoef(f, v)[0, 1]
    return E, r


def run(regime, eps, label):
    reg = dict(regime, seed_eps=eps)
    sim = CellSimulation(gd.make_config(reg, 1), config_name='fluxresp')
    rng = np.random.default_rng(1)
    sim.cells = gd.seeded_colony(reg, MODE, rng)
    gd.set_phenotype(sim.cells, reg)
    gd.equilibrate_nutrient(sim, reg)
    center = np.array([reg['L'] / 2, reg['L'] / 2])
    a = reg['cell_r']

    def front():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        keep = main_cluster_mask(pos, rad)
        return front_radii(pos[keep], center, N_BINS)

    R0 = front()
    c0, grad0 = sample_ahead(sim, center, R0, ahead=1.5 * a)
    for _ in range(STEPS):
        gd.set_phenotype(sim.cells, reg)
        sim._simulation_step()
    R1 = front()
    V = (R1 - R0) / (STEPS * reg['dt'])

    Ec, rc = elasticity(c0, V)
    Eg, rg = elasticity(grad0, V)
    amp = 0.5 * (R0.max() - R0.min())
    print(f"\n--- {label} (eps={eps}) ---")
    print(f"  lobe amplitude {amp:.0f} units = {amp/reg['l_meas']:.1f} x l "
          f"= {amp/a:.0f} cell radii")
    print(f"  front advance V: mean {V.mean():+.4f}, "
          f"tip-vs-valley spread {V.std()/abs(V.mean())*100:.0f}% of mean")
    print(f"  nutrient ahead : varies {(c0.max()-c0.min())/c0.mean()*100:.0f}% "
          f"around the perimeter")
    print(f"  ELASTICITY vs c      E = {Ec:+.3f}   (Pearson r = {rc:+.2f})")
    print(f"  ELASTICITY vs |grad c| E = {Eg:+.3f}   (Pearson r = {rg:+.2f})")
    return dict(label=label, eps=eps, R0=R0, V=V, c=c0, grad=grad0,
                Ec=Ec, rc=rc, Eg=Eg, rg=rg)


def main():
    base = dict(gd.SHORT_L, overlap_iterations=1)
    # measure l once so amplitudes can be quoted in units of it
    probe = CellSimulation(gd.make_config(base, 1), config_name='fluxprobe')
    probe.cells = gd.seeded_colony(dict(base, seed_eps=0.0), 1,
                                   np.random.default_rng(0))
    gd.set_phenotype(probe.cells, base)
    gd.equilibrate_nutrient(probe, base)
    base['l_meas'] = gd.measure_length_scale(probe, base)
    print(f"regime: l = {base['l_meas']:.1f} units "
          f"({base['l_meas']/base['cell_r']:.1f} cell radii)")
    del probe

    runs = [run(base, 0.05, 'small lobes (amplitude < l)'),
            run(base, 0.25, 'large lobes (amplitude >> l)')]

    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 5))
    for ax, r in zip(np.atleast_1d(axes), runs):
        ax.scatter(r['c'], r['V'], s=14, alpha=0.7)
        if np.isfinite(r['Ec']):
            xs = np.linspace(r['c'].min(), r['c'].max(), 20)
            p = np.polyfit(r['c'], r['V'], 1)
            ax.plot(xs, np.polyval(p, xs), 'r-', lw=2)
        ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.set(xlabel='nutrient just ahead of the front',
               ylabel='local front advance V',
               title=f"{r['label']}\nelasticity E = {r['Ec']:+.3f} "
                     f"(r = {r['rc']:+.2f});  M-S needs E ~ 1")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, 'giverso_fluxresponse.png')
    fig.savefig(out, dpi=115)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
