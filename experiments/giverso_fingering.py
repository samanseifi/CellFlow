"""Giverso branching, final attempt: R/l ~ 10 + quiescence + FREE front.

Combines everything the non-dimensional analysis identified as necessary:
  - small l (low nutrient_D) so R/l ~ 10 is reachable at a feasible colony size;
  - quiescence to freeze the depleted interior into a thin active rim;
  - a big domain AND a stop-at-target-radius so the colony halts with a FREE
    front (rim far from the walls) instead of filling the box;
  - low surface tension (weak adhesion, few overlap sweeps) so the thin front
    is free to finger.

Outputs the colony (colored active/quiescent), the front R(theta), and its
angular power spectrum (discrete analog of the paper's dispersion curve; the
dominant nonzero mode = number of fingers).

Run:  python experiments/giverso_fingering.py
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
from experiments.giverso_nondim import front_power_spectrum, measure_diffusion_length  # noqa: E402

L = 600.0
G = 400
TARGET_R = 150.0           # stop here -> free front (walls at 300, gap ~150)
MAX_STEPS = 4000
NUTRIENT_D = 0.02


def config():
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 60,
        'initial_cluster_radius': 10.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 40.0,
        'nutrient_D': NUTRIENT_D, 'chi_nutrient': 0.0,
        'walk_speed': 0.05, 'max_propulsive_force': 2.0,
        'adhesion_strength': 0.0, 'adhesion_cutoff_factor': 1.2,
        'repulsion_strength': 35.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': 15.0, 'overlap_iterations': 1,
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 4,
        'enable_quiescence': True, 'quiescence_nutrient_threshold': 28.0,
        'directed_division': True,
    }


def colony_radius(sim, center):
    pos = np.array([c.position for c in sim.cells])
    return np.max(np.hypot(pos[:, 0] - center, pos[:, 1] - center))


def main():
    sim = CellSimulation(config(), config_name='giverso_fingering')
    center = L / 2
    R = 0.0
    for step in range(MAX_STEPS):
        sim._simulation_step()
        if step % 25 == 0:
            R = colony_radius(sim, center)
            print(f"\rstep {step}  cells={len(sim.cells)}  R={R:.0f}/{TARGET_R:.0f}", end="")
            if R >= TARGET_R:
                break
    print()

    pos = np.array([c.position for c in sim.cells])
    active = np.array([c.active for c in sim.cells])
    cen = np.array([center, center])
    modes, power, Rth = front_power_spectrum(pos, cen, n_bins=180)

    fig = plt.figure(figsize=(16, 5))
    # 1) colony, colored by state
    ax = fig.add_subplot(1, 3, 1)
    for c in sim.cells:
        col = 'limegreen' if c.active else '#243030'
        ax.add_patch(Circle(c.position, c.radius, color=col, lw=0))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title(f"colony: {len(sim.cells)} cells, R={R:.0f}")
    # 2) front R(theta)
    ax = fig.add_subplot(1, 3, 2)
    th = np.linspace(-180, 180, len(Rth))
    ax.plot(th, Rth)
    ax.set(xlabel='angle (deg)', ylabel='front radius', title='front R(theta)')
    ax.grid(True, alpha=0.3)
    # 3) angular power spectrum (dispersion analog)
    ax = fig.add_subplot(1, 3, 3)
    ax.semilogy(modes[1:40], power[1:40] + 1e-9, 'o-')
    dom = 1 + int(np.argmax(power[1:40]))
    ax.set(xlabel='angular mode (# fingers)', ylabel='power',
           title=f'front power spectrum (peak mode = {dom})')
    ax.grid(True, alpha=0.3)
    fig.suptitle("Giverso branching: R/l~10 + quiescence + free front", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'giverso_fingering.png')
    fig.savefig(out, dpi=120)
    ell, _, _ = measure_diffusion_length(NUTRIENT_D, Rc=40.0, iters=800)
    print(f"l~{ell:.0f}, R={R:.0f} -> R/l~{R/ell:.1f}; front roughness std={np.std(Rth):.1f}; "
          f"dominant mode={dom}")
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
