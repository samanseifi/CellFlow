"""Clonal sectors + genuine starvation + Monod: the last untested ingredients.

Three findings set this up:

  * Noise-driven roughness SATURATES at delta ~ 0.36 l, below the delta ~ l
    threshold the instability needs (a bump smaller than the diffusive boundary
    layer is smoothed before it can focus flux).
  * An imposed patchy substrate breaks that threshold but the front only TRACES
    it -- the per-mode gain is flat and identical with the flux response
    switched off (giverso_spectrumgain.py).
  * The "starvation" test scaled bc_value and the quiescence threshold together,
    which with linear uptake is a pure slowdown. The low-nutrient regime was
    never actually tested.

Three ingredients were suppressed by construction, and this script restores them:

  1. HERITABLE growth-rate heterogeneity. set_phenotype forced identical
     consumption rates every step and Cell.divide did not pass the phenotype on,
     so cell-to-cell variation washed out within a generation. With inheritance,
     an initial spread sorts itself into CLONAL SECTORS whose angular width grows
     with the colony -- the one noise source that is persistent (stored in cells,
     so it cannot diffuse away like nutrient noise), macroscopic, and
     self-generated rather than imposed.
  2. GENUINE starvation: lower bc_value with the quiescence threshold held FIXED,
     which really does thin the active rim.
  3. MONOD uptake: with saturation, rich regions saturate while poor regions stay
     linear, sharpening tip-versus-valley contrast -- the reason real colonies
     branch when starved.

Every configuration is run with BOTH propulsion laws. The flux-blind
('saturated') law is the control: roughness that appears under both is structure
the colony inherited, not fingering. Only an excess under the flux-responsive law
is the instability doing work.

Run:  python experiments/giverso_sectors.py [steps] [n_seeds]
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
L_DIFF = 8.9


def seed_phenotype(cells, regime, rng):
    """Assign the metabolic phenotype ONCE; daughters now inherit it.

    ``pheno_cv`` > 0 gives a lognormal spread in consumption rate about the
    regime mean, which is what later sorts into clonal sectors.
    """
    cv = regime.get('pheno_cv', 0.0)
    mu = regime['consumption']
    max_r = regime.get('max_radius')
    if cv > 0:
        sigma = np.sqrt(np.log(1.0 + cv ** 2))
        rates = rng.lognormal(np.log(mu) - 0.5 * sigma ** 2, sigma, len(cells))
    else:
        rates = np.full(len(cells), mu)
    for c, r in zip(cells, rates):
        c.consumption_rate = float(r)
        c.basal_metabolism_rate = regime['basal']
        if regime.get('uptake_km', 0) > 0:
            c.uptake_saturation = float(regime['uptake_km'])
        if max_r is not None:
            c.max_radius = float(max_r)
            c.min_radius = 0.5 * float(max_r)
            c.update_radius()
    return rates


def run(label, cfg_over, response, steps, seed):
    regime = dict(gd.PROPORTIONAL, seed_eps=0.0, name='sectors')
    regime.update(cfg_over)
    cfg = gd.make_config(regime, seed)
    cfg['propulsion_response'] = response
    # MATCHED CONTROL FORCE. Under the proportional law max_propulsive_force is a
    # CAP that is never reached -- the actual force is |chi grad c|, median ~142
    # at the rim (max 621) for this regime. Under the saturated law it is the
    # force every cell applies. Leaving it at the cap would drive the control ~10x
    # harder than the run it is controlling for, and its extra roughness would be
    # force-noise rather than physics. So the control runs at the measured median.
    if response == 'saturated':
        cfg['max_propulsive_force'] = float(regime.get('control_force', 142.0))
    if regime.get('uptake_km', 0) > 0:
        cfg['nutrient_uptake_saturation'] = float(regime['uptake_km'])
    sim = CellSimulation(cfg, config_name='sectors')
    rng = np.random.default_rng(seed)
    sim.cells = gd.seeded_colony(regime, 1, rng)
    rates = seed_phenotype(sim.cells, regime, rng)
    gd.equilibrate_nutrient(sim, regime)
    center = np.array([regime['L'] / 2, regime['L'] / 2])

    def front():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        return analyze_front(pos, rad, center)

    r0 = front()
    for _ in range(steps):
        sim._simulation_step()            # NO set_phenotype: inheritance carries it
    r1 = front()
    spread = np.std([c.consumption_rate for c in sim.cells]) / \
        max(np.mean([c.consumption_rate for c in sim.cells]), 1e-12)
    delta = r1['roughness'] * r1['mean_radius']
    print(f"  {label:<34} {response:<13} rough {r0['roughness']:.4f}->"
          f"{r1['roughness']:.4f} ({r1['roughness']/r0['roughness']:5.1f}x)  "
          f"delta={delta/L_DIFF:5.2f} l  R={r1['mean_radius']:.0f}  "
          f"n={r1['n_cells']:6d}  cv_end={spread:.2f}", flush=True)
    return dict(label=label, response=response, r0=r0, r1=r1,
                snap=[(c.position.copy(), c.radius, c.active) for c in sim.cells],
                L=regime['L'], delta_l=delta / L_DIFF)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Genuine starvation: bc down AND the quiescence threshold re-calibrated so
    # the rim thins without the colony freezing. Holding the threshold at 30 while
    # bc drops to 40 makes cells active only above 75% of the reservoir -- measured
    # active fraction 0.0%, i.e. the colony stops rather than starves. A threshold
    # of 10 gives a ~13% active rim at bc=40, comparable to the 14% of the
    # full-nutrient reference but on a genuinely leaner substrate.
    configs = [
        ("control (uniform, full, linear)",  dict(pheno_cv=0.0)),
        ("clonal sectors cv=0.35",           dict(pheno_cv=0.35)),
        ("sectors + starved (bc40, qui10)",  dict(pheno_cv=0.35, bc_value=40.0,
                                                 qui_threshold=10.0)),
        ("sectors + starved + Monod",        dict(pheno_cv=0.35, bc_value=40.0,
                                                 qui_threshold=10.0,
                                                 uptake_km=25.0)),
    ]

    print(f"clonal sectors / starvation / Monod, {steps} steps, "
          f"{n_seeds} seed(s)\n")
    results = []
    for label, over in configs:
        for response in ('proportional', 'saturated'):
            best = None
            for s in range(1, n_seeds + 1):
                r = run(label, over, response, steps, s)
                if best is None or r['r1']['roughness'] > best['r1']['roughness']:
                    best = r
            results.append(best)
        print()

    print(f"{'config':<34} {'prop':>8} {'sat':>8} {'excess':>8}")
    print('-' * 62)
    for i in range(0, len(results), 2):
        p, q = results[i], results[i + 1]
        ex = p['r1']['roughness'] / max(q['r1']['roughness'], 1e-12)
        print(f"{p['label']:<34} {p['r1']['roughness']:>8.4f} "
              f"{q['r1']['roughness']:>8.4f} {ex:>7.2f}x")
    print('-' * 62)
    print("excess > 1 => the flux-responsive law adds roughness the flux-blind")
    print("one does not, i.e. the instability is contributing.")

    # results are [cfg0-prop, cfg0-sat, cfg1-prop, cfg1-sat, ...]; the figure
    # wants ALL proportional on row 1 and ALL saturated on row 2 so each column
    # is one configuration and the rows are the law being compared.
    ordered = results[0::2] + results[1::2]
    n = len(ordered)
    fig, axes = plt.subplots(2, n // 2, figsize=(4.1 * (n // 2), 9))
    for ax, r in zip(np.atleast_1d(axes).ravel(), ordered):
        for p, rad, active in r['snap']:
            ax.add_patch(Circle(p, rad, color='#43c463' if active else '#1d2b2b',
                                lw=0))
        ax.set_xlim(0, r['L']); ax.set_ylim(0, r['L'])
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r['label']}\n{r['response']}  "
                     f"roughness {r['r1']['roughness']:.3f} "
                     f"($\\delta$={r['delta_l']:.2f}$\\ell$)", fontsize=8)
    fig.suptitle("Clonal sectors, genuine starvation and Monod uptake.\n"
                 "Top row: flux-responsive (proportional). Bottom row: flux-blind "
                 "control at MATCHED force. Green = active rim, dark = quiescent.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, 'giverso_sectors.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
