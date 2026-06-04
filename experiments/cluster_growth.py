"""Cluster growth -> animated GIF, using the real cell-shape PHYSICS.

A small seed grows in a Dirichlet nutrient bath. Fidelity comes from the model
itself (not the renderer):
  - daughters born just touching + multi-sweep overlap resolution (#21) so cells
    don't intertwine during fast division;
  - cell-shape mechanics (#22): each cell carries a deviatoric strain that
    evolves viscoelastically under its contact stress, deforming into an
    area-conserving ellipse (squeezed cells bulge perpendicular; rim cells
    stretch tangentially; isolated cells stay round) -- and relaxes back when
    unloaded. Mechanics stay circular, so it's cheap.

Outputs:
  cluster_growth_simulation.gif
  experiments/cluster_growth_montage.png

Run:  python experiments/cluster_growth.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 85.0,
        'nutrient_D': 0.6, 'chi_nutrient': 6.0,
        'walk_speed': 0.25, 'max_propulsive_force': 12.0,
        'viscosity': 100.0, 'adhesion_strength': 0.4, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 80.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'seed': 7,
        'fluid_model': 'brinkman_fft', 'brinkman_screening_length': 15.0,
        'overlap_iterations': 10, 'growth_model': 'area_conserving',
        # real cell-shape mechanics (stiff: subtle deformation)
        'enable_cell_shape': True, 'shape_compliance': 0.012,
        'shape_relaxation_time': 0.6, 'shape_max_aspect': 1.4,
    }


def _inscribed(a, b, r):
    """Scale ellipse so its long axis = collision radius (no poke past r)."""
    s = r / max(a, b)
    return a * s, b * s


def main():
    sim = CellSimulation(config(), config_name='cluster_growth')
    frames, snaps, counts = [], {}, []
    for step in range(STEPS):
        sim._simulation_step()
        counts.append(len(sim.cells))
        if step % SAVE_EVERY == 0 or step == STEPS - 1:
            frames.append(visualization.render_frame(
                sim.nutrient_field, sim.cells, sim.physical_size, step, sim.output_dir))
        if step in MONTAGE_STEPS:
            # snapshot cell shapes (copy state needed for the montage)
            snaps[step] = (sim.nutrient_field.copy(),
                           [(c.position.copy(), c.radius, c.exx, c.exy, c.phase)
                            for c in sim.cells])
        print(f"\rstep {step+1}/{STEPS}  cells={len(sim.cells)}", end="")
    print()

    fig, axes = plt.subplots(1, len(MONTAGE_STEPS), figsize=(4 * len(MONTAGE_STEPS), 4.2))
    for ax, step in zip(axes, MONTAGE_STEPS):
        nut, cells = snaps[step]
        ax.imshow(nut, cmap='viridis', origin='lower', extent=[0, L, 0, L])
        for pos, r, exx, exy, ph in cells:
            m = np.hypot(exx, exy)
            a, b = _inscribed(r * np.exp(m), r * np.exp(-m), r)
            ang = np.degrees(0.5 * np.arctan2(exy, exx))
            ax.add_patch(Ellipse(pos, 2 * a, 2 * b, angle=ang,
                                 color='red' if ph == 'DIVISION' else 'white',
                                 alpha=0.85, ec='black', lw=0.4))
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}   cells={len(cells)}", fontsize=10)
    fig.suptitle("Cluster growth with viscoelastic cell-shape mechanics "
                 "(area-conserving ellipses from contact stress)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    montage = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'cluster_growth_montage.png')
    fig.savefig(montage, dpi=110)
    print(f"cells: {counts[0]} -> {counts[-1]}")
    print(f"Saved montage -> {montage}")

    visualization.create_gif(frames, sim.config_name)
    print(f"Saved GIF -> {sim.config_name}_simulation.gif")


if __name__ == '__main__':
    main()
