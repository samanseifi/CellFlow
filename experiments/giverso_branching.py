"""Discrete replication of Giverso, Verani & Ciarletta (2015), "Emerging
morphologies in round bacterial colonies: comparing volumetric versus
chemotactic expansion" (Biomech. Model. Mechanobiol.).

The continuum paper shows that a round colony expanding on agar develops a
linearly unstable, branched front, and that VOLUMETRIC (mitotic source) and
CHEMOTACTIC (non-convective flux) expansion give qualitatively different
morphologies. Here we reproduce the phenomenon with the discrete agent model:
nutrient is supplied from the dish edges (Dirichlet reservoir) and consumed by
the colony, so a depletion gradient forms and the growing front is
diffusion-limited (Mullins--Sekerka / DLA-type instability) -> fingering.

  - volumetric:  growth + division drive expansion (chi ~ 0). Protrusions reach
                 fresher nutrient, divide faster, and amplify -> branches.
  - chemotactic: cells migrate up the nutrient gradient (large chi), with little
                 division -> chemotactic fingering.

Usage:
  python experiments/giverso_branching.py volumetric
  python experiments/giverso_branching.py chemotactic
  python experiments/giverso_branching.py compare      # both, side by side
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation       # noqa: E402

L = 1000.0
G = 500
STEPS = 1800


def config(mode):
    base = {
        'initial_setup_type': 'central_uniform', 'num_cells': 40,
        'initial_cluster_radius': 7.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 50.0,
        'nutrient_D': 0.03,                 # strongly diffusion-limited front
        # weak adhesion = low surface tension, so the branching instability is
        # not smoothed away (surface tension is the stabilizing term).
        'adhesion_strength': 0.015, 'adhesion_cutoff_factor': 1.3,
        'repulsion_strength': 80.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': 12.0, 'overlap_iterations': 4,
        'growth_model': 'area_conserving', 'enable_visualization': False,
        'enable_quiescence': True, 'quiescence_nutrient_threshold': 26.0,
        'seed': 11,
    }
    if mode == 'volumetric':
        base.update({'chi_nutrient': 0.0, 'walk_speed': 0.05,
                     'max_propulsive_force': 4.0})
    elif mode == 'chemotactic':
        base.update({'chi_nutrient': 30.0, 'walk_speed': 0.3,
                     'max_propulsive_force': 25.0})
    else:
        raise ValueError(mode)
    return base


def run(mode):
    sim = CellSimulation(config(mode), config_name=f'giverso_{mode}')
    for step in range(STEPS):
        sim._simulation_step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"\r[{mode}] step {step+1}/{STEPS}  cells={len(sim.cells)}", end="")
    print()
    return sim


def draw(ax, sim, title):
    ax.imshow(sim.nutrient_field, cmap='bone', origin='lower', extent=[0, L, 0, L])
    for c in sim.cells:
        col = 'limegreen' if getattr(c, 'active', True) else '#243030'
        ax.add_patch(Circle(c.position, c.radius, color=col, alpha=0.95, lw=0))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}  ({len(sim.cells)} cells)", fontsize=11)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'volumetric'
    if mode == 'compare':
        fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
        for a, m in zip(ax, ['volumetric', 'chemotactic']):
            draw(a, run(m), f"{m} expansion")
        fig.suptitle("Discrete replication of Giverso et al. (2015): "
                     "volumetric vs chemotactic colony branching", fontsize=12)
        out = 'giverso_compare.png'
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        draw(ax, run(mode), f"{mode} expansion")
        out = f'giverso_{mode}.png'
    fig.tight_layout()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out)
    fig.savefig(path, dpi=120)
    print(f"Saved -> {path}")


if __name__ == '__main__':
    main()
