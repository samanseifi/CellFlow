"""Seed-averaged wound closure vs hydrodynamic coupling range (rigorous).

Builds on wound_closure_isolated.py (which holds single-cell speed FIXED across
screening lengths so only the coupling RANGE varies) and adds what issue #11
now makes possible: ENSEMBLE averaging. Each (delta) is run over several seeds
with a nonzero random walk, and we report mean +/- std of the closure time and
the front roughness. Error bars tell us whether the coupling-range effect is
real or within run-to-run noise.

Requires the reproducibility fix (#11): config['seed'] makes each run
deterministic, so the ensemble is controlled and repeatable.

Produces experiments/wound_closure_ensemble.png and a table.
Run:  python experiments/wound_closure_ensemble.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402
from experiments.wound_closure_isolated import (  # noqa: E402
    self_mobility_factor, centerline_ridge, coverage_grid, left_front,
    fingering_wavelength, L, G, GAP, SPACING, X0, X1, F_REF,
)

DELTAS = [8.0, 16.0, 32.0, 64.0]
SEEDS = [0, 1, 2, 3, 4]
STEPS = 250
COVER_THRESH = 0.50
WALK = 1.5
TARGET_SPEED = 12.0       # fixed single-cell speed (sets mu per delta)


def config(mu, delta, seed):
    return {
        'initial_setup_type': 'wound_healing',
        'wound_gap_width': GAP, 'initial_cell_spacing': SPACING,
        'dt': 0.01, 'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_value': 15.0, 'nutrient_D': 0.5, 'chi_nutrient': 12.0,
        'walk_speed': WALK, 'max_propulsive_force': F_REF, 'viscosity': mu,
        'adhesion_strength': 0.3, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': delta, 'seed': seed,
    }


def run(mu, delta, seed):
    sim = CellSimulation(config(mu, delta, seed), config_name=f'ens_d{delta:g}_s{seed}')
    ridge = centerline_ridge()
    closure_step = None
    for step in range(STEPS):
        sim.nutrient_field = ridge.copy()
        sim._simulation_step()
        if step % 5 == 0 or step == STEPS - 1:
            if closure_step is None and coverage_grid(sim)[:, X0:X1].mean() >= COVER_THRESH:
                closure_step = step
                break
    _, rough = fingering_wavelength(left_front(coverage_grid(sim)))
    return (closure_step if closure_step is not None else STEPS), rough


def main():
    # Calibrate mu(delta) for a fixed single-cell speed (isolate coupling range).
    mu_of = {d: self_mobility_factor(d) / TARGET_SPEED for d in DELTAS}
    print(f"target single-cell speed v0 = {TARGET_SPEED}")
    print(f"{'delta':>6} {'mu':>7} {'closure mean+-std':>20} {'roughness mean+-std':>22}")

    closures_m, closures_s, rough_m, rough_s = [], [], [], []
    for d in DELTAS:
        cs, rs = [], []
        for s in SEEDS:
            c, r = run(mu_of[d], d, s)
            cs.append(c); rs.append(r)
        cm, csd = np.mean(cs), np.std(cs)
        rm, rsd = np.mean(rs), np.std(rs)
        closures_m.append(cm); closures_s.append(csd)
        rough_m.append(rm); rough_s.append(rsd)
        print(f"{d:>6.0f} {mu_of[d]:>7.3f} {cm:>10.1f} +- {csd:>5.1f} "
              f"{rm:>12.2f} +- {rsd:>5.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].errorbar(DELTAS, closures_m, yerr=closures_s, fmt='o-', capsize=4)
    ax[0].set(xlabel='screening length delta (coupling range)',
              ylabel='closure time (steps)',
              title=f'Closure time vs coupling range\n(single-cell speed FIXED, '
                    f'{len(SEEDS)} seeds, mean +/- std)')
    ax[1].errorbar(DELTAS, rough_m, yerr=rough_s, fmt='s-', color='crimson', capsize=4)
    ax[1].set(xlabel='screening length delta (coupling range)',
              ylabel='front roughness (std)',
              title='Front fingering vs coupling range')
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle("Seed-averaged hydrodynamic wound closure", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'wound_closure_ensemble.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
