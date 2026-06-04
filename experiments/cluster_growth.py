"""Cluster growth simulation -> animated GIF.

A small seed of cells sits in a nutrient bath (Dirichlet reservoir at the
domain edges). Cells consume nutrient, grow, and divide, so the colony expands;
as the interior depletes, a nutrient gradient forms and growth concentrates at
the rim. Uses the Brinkman fluid (gentle) so it stays fast.

Outputs (in the repo root / experiments/):
  cluster_growth_simulation.gif   - the animation
  experiments/cluster_growth_montage.png - snapshot montage (for quick view)

Run:  python experiments/cluster_growth.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation       # noqa: E402
from cellflow import visualization                    # noqa: E402

L = 100.0
G = 100
STEPS = 400
SAVE_EVERY = 8
MONTAGE_STEPS = [0, 90, 180, 280, STEPS - 1]


def config():
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 6,
        'initial_cluster_radius': 4.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 100.0,
        'nutrient_D': 0.6, 'chi_nutrient': 8.0,
        'walk_speed': 0.3, 'max_propulsive_force': 15.0,
        'viscosity': 100.0, 'adhesion_strength': 0.4, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 60.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'seed': 7,
        'fluid_model': 'brinkman_fft', 'brinkman_screening_length': 15.0,
    }


def main():
    sim = CellSimulation(config(), config_name='cluster_growth')
    frames = []
    snaps = {}
    counts = []
    for step in range(STEPS):
        sim._simulation_step()
        counts.append(len(sim.cells))
        if step % SAVE_EVERY == 0 or step == STEPS - 1:
            frames.append(visualization.render_frame(
                sim.nutrient_field, sim.cells, sim.physical_size, step, sim.output_dir))
        if step in MONTAGE_STEPS:
            snaps[step] = (sim.nutrient_field.copy(),
                           np.array([c.position for c in sim.cells]),
                           np.array([c.radius for c in sim.cells]),
                           [c.phase for c in sim.cells])
        print(f"\rstep {step+1}/{STEPS}  cells={len(sim.cells)}", end="")
    print()

    # montage
    fig, axes = plt.subplots(1, len(MONTAGE_STEPS), figsize=(4 * len(MONTAGE_STEPS), 4.2))
    for ax, step in zip(axes, MONTAGE_STEPS):
        nut, pos, rad, phase = snaps[step]
        ax.imshow(nut, cmap='viridis', origin='lower', extent=[0, L, 0, L])
        for p, r, ph in zip(pos, rad, phase):
            ax.add_artist(plt.Circle(p, r, color='red' if ph == 'DIVISION' else 'white',
                                     alpha=0.85))
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}   cells={len(pos)}", fontsize=10)
    fig.suptitle("Cluster growth: nutrient-fed colony expansion (Dirichlet bath)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    montage = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'cluster_growth_montage.png')
    fig.savefig(montage, dpi=110)
    print(f"cells: {counts[0]} -> {counts[-1]}")
    print(f"Saved montage -> {montage}")

    # GIF (deletes the intermediate frames)
    visualization.create_gif(frames, sim.config_name)
    print(f"Saved GIF -> {sim.config_name}_simulation.gif")


if __name__ == '__main__':
    main()
