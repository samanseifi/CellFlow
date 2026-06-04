"""ECM in a real problem: a growing colony walls itself off hydrodynamically.

A colony grows, divides, and deposits ECM. The accumulated matrix (densest in
the core) raises the local Brinkman drag, screening the flow the colony itself
generates. We run the real colony, then solve the fluid for the SAME cell
forces two ways -- with the cell-built ECM drag field, and with uniform base
drag (the counterfactual "no matrix") -- to isolate what the matrix does.

Produces experiments/ecm_colony.png.
Run:  python experiments/ecm_colony.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                       # noqa: E402
from cellflow.fluid.brinkman_fft import solve_velocity, solve_velocity_variable_alpha  # noqa: E402
from cellflow.fluid.ibm import spread_forces_blob_numba              # noqa: E402

L = 100.0
G = 100
STEPS = 160


def config():
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 8,
        'initial_cluster_radius': 5.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 80.0,
        'nutrient_D': 0.6, 'chi_nutrient': 5.0,
        'walk_speed': 0.0, 'max_propulsive_force': 12.0,
        'viscosity': 5.0, 'adhesion_strength': 0.4, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 70.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'seed': 5,
        'fluid_model': 'brinkman_fft', 'brinkman_screening_length': 25.0,
        'overlap_iterations': 8, 'growth_model': 'area_conserving',
        'enable_ecm': True, 'ecm_secretion_rate': 0.8,
        'ecm_decay_rate': 0.03, 'ecm_drag_coeff': 0.6,
    }


def main():
    sim = CellSimulation(config(), config_name='ecm_colony')
    for step in range(STEPS):
        sim._simulation_step()
        print(f"\rstep {step+1}/{STEPS}  cells={len(sim.cells)}", end="")
    print()

    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])
    # same cell forces for both fluid solves
    prop, mono = sim._calculate_forces(pos, rad)
    total = prop + mono
    sigmas = sim.ibm_reg_factor * rad
    fd = spread_forces_blob_numba(pos, total, sigmas, G, G, sim.dx)

    base = sim.brinkman_alpha
    alpha_ecm = base + sim.ecm_drag_coeff * sim.ecm_field
    u_no, _ = solve_velocity(fd, mu=sim.viscosity, dx=sim.dx, alpha=base), None
    u_ecm, iters, res = solve_velocity_variable_alpha(
        fd, mu=sim.viscosity, dx=sim.dx, alpha_field=alpha_ecm)
    s_no = np.hypot(u_no[:, :, 0], u_no[:, :, 1])
    s_ecm = np.hypot(u_ecm[:, :, 0], u_ecm[:, :, 1])

    # mean flow inside the colony (within its radius of gyration)
    com = pos.mean(0)
    rg = np.sqrt(np.mean(np.sum((pos - com) ** 2, 1)))
    yy, xx = np.mgrid[0:G, 0:G] * sim.dx
    core = ((xx - com[0]) ** 2 + (yy - com[1]) ** 2) <= rg ** 2
    mean_no = s_no[core].mean()
    mean_ecm = s_ecm[core].mean()

    vmax = max(s_no.max(), s_ecm.max())
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(sim.ecm_field, cmap='YlOrBr', origin='lower', extent=[0, L, 0, L])
    ax[0].scatter(pos[:, 0], pos[:, 1], s=6, c='steelblue', alpha=0.6)
    ax[0].set_title(f"Cell-deposited ECM ({len(pos)} cells)\nalpha {alpha_ecm.min():.2f}-{alpha_ecm.max():.2f}")
    for a, s, ttl in [(ax[1], s_no, f"Flow WITHOUT matrix\nmean core speed={mean_no:.3f}"),
                      (ax[2], s_ecm, f"Flow WITH cell-built matrix\nmean core speed={mean_ecm:.3f}")]:
        im = a.imshow(s, cmap='viridis', origin='lower', extent=[0, L, 0, L], vmax=vmax)
        a.streamplot(np.arange(G) * sim.dx, np.arange(G) * sim.dx,
                     (u_no if s is s_no else u_ecm)[:, :, 0],
                     (u_no if s is s_no else u_ecm)[:, :, 1],
                     color='white', density=1.0, linewidth=0.6, arrowsize=0.7)
        a.set_title(ttl)
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, L); a.set_ylim(0, L)
    fig.suptitle(f"A growing colony screens its own flow with deposited ECM "
                 f"(core flow {mean_no:.3f} -> {mean_ecm:.3f}, {100*(1-mean_ecm/mean_no):.0f}% reduction)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecm_colony.png')
    fig.savefig(out, dpi=110)
    print(f"core flow: {mean_no:.4f} (no matrix) -> {mean_ecm:.4f} (with matrix), "
          f"solver {iters} iters res={res:.1e}")
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
