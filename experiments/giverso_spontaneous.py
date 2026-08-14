"""Does the colony finger ON ITS OWN, from cell-scale noise?

Every lobed picture so far came from a SEEDED perturbation being amplified. That
shows the instability exists, but Giverso's fingers appear spontaneously. This is
the test that distinguishes "we can amplify what we seed" from "the colony
fingers by itself".

Free growth (no seeded mode) in the flux-proportional regime, run long enough for
cell-scale noise to reach visible amplitude. The measured growth rate
(lambda ~ 0.024 at the selected mode) and the noise floor (a_k ~ 5e-3 after 1000
steps) put that at a few thousand steps.

The important output is not the final picture but the SPECTRUM OVER TIME: if the
front is genuinely unstable with a selected wavelength, the mode amplitudes should
grow exponentially at their own measured rates and the spectrum should develop a
peak near k*, not amplify uniformly or stay flat.

Run:  python experiments/giverso_spontaneous.py [steps] [seed]
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
from cellflow.analysis.front import (analyze_front, fit_growth_rate)  # noqa: E402
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_EVERY = 250
KMAX = 25


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Box sized for the FINAL colony, not the initial one. R grows at
    # ~0.00165/time here, so an 8000-step run takes R 360 -> ~700; in the 2x
    # box (wall at 800) that would run into the Dirichlet nutrient boundary and
    # contaminate the result.
    regime = dict(gd.PROPORTIONAL_BIG, name='spontaneous', seed_eps=0.0,
                  L=2000.0, G=1000)
    sim = CellSimulation(gd.make_config(regime, seed), config_name='spontaneous')
    sim.cells = gd.seeded_colony(regime, 1, np.random.default_rng(seed))
    gd.set_phenotype(sim.cells, regime)
    gd.equilibrate_nutrient(sim, regime)
    center = np.array([regime['L'] / 2, regime['L'] / 2])

    def state():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        return analyze_front(pos, rad, center)

    times, spectra, snaps = [], [], []

    def sample(step):
        res = state()
        times.append(step * regime['dt'])
        spectra.append(res['modes'][:KMAX + 1].copy())
        dom = int(np.argmax(res['modes'][2:KMAX + 1])) + 2
        print(f"  step {step:5d}  n={res['n_cells']:6d}  R={res['mean_radius']:6.1f}  "
              f"rough={res['roughness']:.4f}  dominant k={dom:2d} "
              f"(a={res['modes'][dom]:.4f})", flush=True)
        return res

    print(f"free growth, {steps} steps, seed {seed} "
          f"(flux-proportional, R0={regime['R0']:.0f})", flush=True)
    sample(0)
    snaps.append((0, [(c.position.copy(), c.radius, c.active) for c in sim.cells]))
    for step in range(steps):
        gd.set_phenotype(sim.cells, regime)
        sim._simulation_step()
        if (step + 1) % SAMPLE_EVERY == 0:
            sample(step + 1)
            # checkpoint: keep the latest snapshot and redraw, so a long run can
            # be inspected (or stopped) at any point without losing everything
            snaps[-1:] = [(step + 1, [(c.position.copy(), c.radius, c.active)
                                      for c in sim.cells])]
            if len(snaps) < 2:
                snaps.append(snaps[-1])
            try:
                render(times, spectra, snaps, regime)
            except Exception as exc:      # never let plotting kill the run
                print(f"    (checkpoint render skipped: {exc})", flush=True)
        if (step + 1) == steps // 2:
            snaps.insert(1, (step + 1,
                             [(c.position.copy(), c.radius, c.active)
                              for c in sim.cells]))

    render(times, spectra, snaps, regime)


def render(times, spectra, snaps, regime):
    t = np.array(times)
    S = np.array(spectra)                       # (n_times, KMAX+1)

    # per-mode growth rate from the noise-driven run
    print("\n  per-mode growth rate measured from this free-growth run:")
    rates = {}
    for k in range(2, KMAX + 1):
        fit = fit_growth_rate(t, S[:, k])
        rates[k] = fit['lambda_']
        if k <= 14:
            print(f"    k={k:2d}: lambda={fit['lambda_']:+.4f} "
                  f"(r2={fit['r_squared']:.2f})", flush=True)
    best = max(rates, key=lambda k: rates[k] if np.isfinite(rates[k]) else -9)
    print(f"  fastest-growing mode from noise: k = {best}", flush=True)

    fig = plt.figure(figsize=(16, 9))
    for i, (step, snap) in enumerate(snaps):
        ax = fig.add_subplot(2, 3, i + 1)
        for p, r, active in snap:
            ax.add_patch(Circle(p, r, color='#43c463' if active else '#1d2b2b', lw=0))
        ax.set_xlim(0, regime['L']); ax.set_ylim(0, regime['L'])
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}", fontsize=10)

    ax = fig.add_subplot(2, 3, 4)
    for j in range(0, len(t), max(1, len(t) // 6)):
        ax.plot(range(2, KMAX + 1), S[j, 2:], 'o-', ms=3,
                label=f't={t[j]:.0f}')
    ax.set(xlabel='mode k', ylabel='amplitude $a_k$', yscale='log',
           title='front spectrum over time')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    for k in (2, 4, 6, 8, 10, 14):
        ax.semilogy(t, S[:, k], 'o-', ms=3, label=f'k={k}')
    ax.set(xlabel='time', ylabel='$a_k$',
           title='mode amplitudes (straight = exponential)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 6)
    ks = sorted(rates)
    ax.plot(ks, [rates[k] for k in ks], 'o-')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set(xlabel='mode k', ylabel=r'$\lambda$ from noise',
           title='dispersion recovered from NOISE alone')
    ax.grid(alpha=0.3)

    fig.suptitle("Spontaneous fingering test: free growth from cell-scale noise, "
                 "flux-proportional chemotaxis", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, 'giverso_spontaneous.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
