"""Heterogeneous nutrient + starvation: can a patchy substrate seed the fingering?

The spontaneous test (giverso_spontaneous.py) showed noise-driven front roughness
SATURATES: individual mode amplitudes plateau at delta ~ 0.36 l, while every
seeded mode that actually amplified had delta >~ 2 l. The instability is
subcritical -- a bump smaller than the diffusive boundary layer is smoothed over
before it can focus flux, so cell-scale noise (delta ~ one cell radius, ~4x below
threshold) can never bootstrap itself.

The nutrient field in every run so far has been perfectly smooth: Dirichlet
boundaries hold it uniform, and the equilibration builds a clean radial
exponential. The only structure in it is the colony's own uptake. So the field
that DRIVES the instability carries no noise at all.

This script gives the substrate spatial structure -- a correlated Gaussian random
field of relative amplitude A and correlation length xi -- and optionally starves
the colony. A rich patch lets the front run ahead locally; if that displacement
reaches the delta ~ l threshold, the instability should take over and keep going.

Two things to keep honest:
  * Nutrient DIFFUSES, so the patchiness decays on xi^2/D. At xi = 2l that is
    ~1300 steps, comparable to the run; at xi = l it washes out in ~300. The
    perturbation therefore has a limited window to reach threshold.
  * Starving is not free: the quiescence threshold has to scale with the nutrient
    level or the whole colony simply goes passive and nothing grows at all.

Run:  python experiments/giverso_heterogeneous.py [steps]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.analysis.front import analyze_front                   # noqa: E402
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_EVERY = 200


def correlated_field(shape, dx, corr_length, rng):
    """Zero-mean, unit-variance Gaussian random field with the given correlation
    length (white noise smoothed by a Gaussian kernel in Fourier space)."""
    w = rng.standard_normal(shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=dx)
    KX, KY = np.meshgrid(kx, ky)
    filt = np.exp(-0.5 * (KX ** 2 + KY ** 2) * corr_length ** 2)
    f = np.fft.ifft2(np.fft.fft2(w) * filt).real
    s = f.std()
    return f / s if s > 0 else f


def run(label, amplitude, corr_length, bc_scale, steps, seed=1):
    """One colony. ``bc_scale`` < 1 starves it (quiescence threshold scaled with
    the nutrient level, else the whole colony goes passive)."""
    regime = dict(gd.PROPORTIONAL, seed_eps=0.0, name='hetero')
    regime['bc_value'] = gd.PROPORTIONAL['bc_value'] * bc_scale
    regime['qui_threshold'] = gd.PROPORTIONAL['qui_threshold'] * bc_scale

    sim = CellSimulation(gd.make_config(regime, seed), config_name='hetero')
    rng = np.random.default_rng(seed)
    sim.cells = gd.seeded_colony(regime, 1, rng)
    gd.set_phenotype(sim.cells, regime)
    gd.equilibrate_nutrient(sim, regime)
    center = np.array([regime['L'] / 2, regime['L'] / 2])

    if amplitude > 0.0:
        noise = correlated_field(sim.nutrient_field.shape, sim.dx,
                                 corr_length, rng)
        sim.nutrient_field *= (1.0 + amplitude * noise)
        np.clip(sim.nutrient_field, 0.0, None, out=sim.nutrient_field)

    def front():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        return analyze_front(pos, rad, center)

    res = front()
    trace = [(0, res['roughness'], res['mean_radius'], res['n_cells'])]
    for step in range(steps):
        gd.set_phenotype(sim.cells, regime)
        sim._simulation_step()
        if (step + 1) % SAMPLE_EVERY == 0:
            res = front()
            trace.append((step + 1, res['roughness'], res['mean_radius'],
                          res['n_cells']))

    final = front()
    l = 8.9
    delta = final['roughness'] * final['mean_radius']
    print(f"  {label:<34} rough {trace[0][1]:.4f} -> {final['roughness']:.4f} "
          f"({final['roughness']/trace[0][1]:5.2f}x)  delta={delta:5.1f} "
          f"({delta/l:4.2f} l)  R={final['mean_radius']:.0f}  "
          f"n={final['n_cells']:6d}", flush=True)
    snap = [(c.position.copy(), c.radius, c.active) for c in sim.cells]
    return dict(label=label, trace=trace, final=final, snap=snap,
                L=regime['L'])


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    l = 8.9
    print(f"heterogeneous-substrate test, {steps} steps "
          f"(l = {l:.1f}; noise-plateau delta was 0.36 l, amplifying modes >= 2 l)\n")

    cases = [
        ("control: smooth, full nutrient",      0.0, 0.0,    1.00),
        ("noise A=0.3, xi=2l, full nutrient",   0.3, 2 * l,  1.00),
        ("noise A=0.6, xi=2l, full nutrient",   0.6, 2 * l,  1.00),
        ("noise A=0.6, xi=4l, full nutrient",   0.6, 4 * l,  1.00),
        ("control: smooth, STARVED (0.3x)",     0.0, 0.0,    0.30),
        ("noise A=0.6, xi=2l, STARVED (0.3x)",  0.6, 2 * l,  0.30),
        ("noise A=0.6, xi=4l, STARVED (0.3x)",  0.6, 4 * l,  0.30),
    ]
    results = [run(lab, a, xi, bc, steps) for lab, a, xi, bc in cases]

    n = len(results)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.1 * ((n + 1) // 2), 9))
    for ax, r in zip(np.atleast_1d(axes).ravel(), results):
        for p, rad, active in r['snap']:
            ax.add_patch(Circle(p, rad, color='#43c463' if active else '#1d2b2b',
                                lw=0))
        ax.set_xlim(0, r['L']); ax.set_ylim(0, r['L'])
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        d = r['final']['roughness'] * r['final']['mean_radius']
        ax.set_title(f"{r['label']}\nroughness {r['final']['roughness']:.3f}"
                     f"  ($\\delta$={d/8.9:.2f}$\\ell$)", fontsize=8)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis('off')
    fig.suptitle("Heterogeneous nutrient substrate: does patchy food seed the "
                 "instability past its threshold?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, 'giverso_heterogeneous.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
