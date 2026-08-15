"""Starvation to the freezing point: does the front ever branch?

Why this is worth running even though the dispersion relation says it will not.
Every previous starvation test in this study -- the bc=40/qui=10 run in
giverso_sectors.py, and the 0.3x scaling before it -- was run on the BROKEN
nutrient field: the exterior was seeded at the reservoir value and could not
relax, so the colony edge sat at 0.66 n_bc instead of the quasi-steady 0.05
(see docs/giverso_replication.md, "the exterior nutrient field was never at
steady state"). Lowering ``bc_value`` on that field lowered a number that was
not controlling anything. So the low-nutrient limit has still not actually been
probed, and it is cheap to probe properly now.

What "to the limit" means here
------------------------------
NOT scaling bc_value and the quiescence threshold together -- with linear uptake
that is a pure rate change, which is the error the doc already records twice.
The genuine knob is bc_value DOWN with the quiescence threshold held FIXED in
absolute terms. That really does thin the active rim, and it has a hard end: the
rim vanishes and the colony freezes when the interface nutrient falls below the
threshold. This script walks bc down to that freeze and reports where it is,
rather than stopping at a hand-picked value.

The control that makes the answer readable
------------------------------------------
Roughness alone cannot distinguish an unstable front from a colony that merely
grows faster where there is more food -- giverso_spectrumgain.py established
that. So every configuration runs under BOTH propulsion laws with the control
force matched, and the reported quantity is the EXCESS of the flux-responsive
law over the flux-blind one. Excess > 1 is the instability doing work; excess
<= 1 means whatever roughness appeared is tracing, not fingering.

Reported alongside: delta/l, the lobe depth in diffusion lengths. Every seeded
mode that ever amplified in this study had delta >~ 2 l; noise-driven modes
plateau near 0.36 l. That threshold is what a branching run has to clear.

Run:  python experiments/giverso_starvation_limit.py [steps] [seeds]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.analysis.front import analyze_front                   # noqa: E402
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# On the corrected field the interface sits at ~0.046 n_bc, so the reference
# regime's threshold of 1.4 is ~30% of the interface value. Holding 1.4 FIXED
# while bc falls is the genuine starvation axis: the rim thins because there is
# less nutrient at the front, not because the criterion moved with it.
QUI_FIXED = 1.4

# bc_value ladder. The reference regime is 100; the interface scales with it, so
# the rim should vanish somewhere near bc ~ 32 (interface 0.046*32 = 1.47, just
# above the threshold). The ladder is deliberately taken past that point so the
# freeze is MEASURED rather than assumed -- and so the last living colony is the
# genuine limit, not a value chosen because it looked good.
BC_LADDER = [100.0, 70.0, 55.0, 45.0, 40.0, 36.0, 33.0, 30.0, 27.0, 24.0, 21.0]


def run(bc, response, steps, seed, control_force):
    regime = dict(gd.PROPORTIONAL_FIXED, name='starvation_limit',
                  bc_value=bc, qui_threshold=QUI_FIXED, seed_eps=0.0)
    cfg = gd.make_config(regime, seed)
    cfg['propulsion_response'] = response
    if response == 'saturated':
        cfg['max_propulsive_force'] = control_force
    sim = CellSimulation(cfg, config_name='starvation_limit')
    rng = np.random.default_rng(seed)
    sim.cells = gd.seeded_colony(regime, 1, rng)
    gd.set_phenotype(sim.cells, regime)
    eq_it, eq_ok = gd.equilibrate_nutrient(sim, regime)
    l_meas = gd.measure_length_scale(sim, regime)
    center = np.array([regime['L'] / 2, regime['L'] / 2])

    def front():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        return analyze_front(pos, rad, center)

    r0 = front()
    for _ in range(steps):
        sim._simulation_step()
    r1 = front()

    active = float(np.mean([c.active for c in sim.cells]))
    steady = gd.check_exterior_steady(sim, regime, r1['mean_radius'])
    delta = r1['roughness'] * r1['mean_radius']
    grew = r1['mean_radius'] / r0['mean_radius']
    frozen = bool(active < 0.01 or grew < 1.002)
    return dict(bc=bc, response=response, seed=seed, l=l_meas,
                rough0=r0['roughness'], rough1=r1['roughness'],
                R0=r0['mean_radius'], R1=r1['mean_radius'],
                n0=r0['n_cells'], n1=r1['n_cells'],
                active=active, delta_l=delta / max(l_meas, 1e-9),
                frozen=frozen, eq_ok=eq_ok, exterior_steady=steady['steady'],
                snap=[(c.position.copy(), c.radius, c.active) for c in sim.cells],
                L=regime['L'])


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Matched control force: under the proportional law max_propulsive_force is a
    # cap that is never reached, so leaving the flux-blind control at the cap
    # would drive it ~10x harder than the run it controls for. giverso_sectors.py
    # measured the median proportional force at 142 for this regime; the drive is
    # |chi grad c| and grad c scales with bc, so the matched force scales too.
    print(f"Starvation to the freezing point, on the CORRECTED nutrient field.\n"
          f"{steps} steps, {seeds} seed(s), quiescence threshold held at "
          f"{QUI_FIXED} throughout.\n")
    print(f"{'bc':>6}{'active':>8}{'l':>6}{'R0->R1':>16}{'cells':>14}"
          f"{'rough':>18}{'delta/l':>9}{'state':>9}")
    print('-' * 92)

    results = []
    for bc in BC_LADDER:
        force = 142.0 * (bc / 100.0)
        row = {}
        for response in ('proportional', 'saturated'):
            best = None
            for s in range(1, seeds + 1):
                r = run(bc, response, steps, s, force)
                if best is None or r['rough1'] > best['rough1']:
                    best = r
            row[response] = best
            results.append(best)
        p, q = row['proportional'], row['saturated']
        state = 'FROZEN' if p['frozen'] else ('ok' if p['exterior_steady'] else 'unsteady')
        print(f"{bc:>6.0f}{p['active']*100:>7.0f}%{p['l']:>6.1f}"
              f"{p['R0']:>7.1f}->{p['R1']:<8.1f}{p['n0']:>6d}->{p['n1']:<7d}"
              f"{p['rough0']:>8.4f}->{p['rough1']:<9.4f}{p['delta_l']:>9.2f}"
              f"{state:>9}")
        if p['frozen']:
            print(f"       ^ colony no longer advancing: this is the limit")
            break

    print('-' * 92)
    print(f"\n{'bc':>6}{'proportional':>14}{'flux-blind':>13}{'excess':>9}"
          f"{'delta/l':>9}")
    print('-' * 52)
    for i in range(0, len(results), 2):
        p, q = results[i], results[i + 1]
        ex = p['rough1'] / max(q['rough1'], 1e-12)
        print(f"{p['bc']:>6.0f}{p['rough1']:>14.4f}{q['rough1']:>13.4f}"
              f"{ex:>8.2f}x{p['delta_l']:>9.2f}")
    print('-' * 52)
    print("excess > 1 : the flux-responsive law adds roughness the flux-blind one")
    print("             does not, i.e. the instability is contributing.")
    print("delta/l >~ 2 : the lobe depth every amplifying seeded mode needed.")

    prop = [r for r in results if r['response'] == 'proportional']
    n = len(prop)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
    for ax, r in zip(np.atleast_1d(axes).ravel(), prop):
        for pos, rad, act in r['snap']:
            ax.add_patch(Circle(pos, rad, color='#43c463' if act else '#1d2b2b',
                                lw=0))
        m = 1.25 * r['R1']
        ax.set_xlim(r['L'] / 2 - m, r['L'] / 2 + m)
        ax.set_ylim(r['L'] / 2 - m, r['L'] / 2 + m)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$n_{{bc}}$={r['bc']:.0f}  active {r['active']*100:.0f}%\n"
                     f"roughness {r['rough1']:.4f}  "
                     f"$\\delta$={r['delta_l']:.2f}$\\ell$", fontsize=8)
    fig.suptitle("Starvation to the freezing point, corrected nutrient field.\n"
                 "Green = active rim, dark = quiescent core. Flux-responsive law.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(HERE, 'giverso_starvation_limit.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")
    return results


if __name__ == '__main__':
    main()
