"""Mechanical feedback on growth: contact inhibition / homeostatic pressure.

Cells exert and feel contact forces, but by default their *biology* ignores
mechanical stress -- a well-fed colony just keeps dividing into an ever denser,
overlapping mass. Real tissues don't: cells under compression stop proliferating
(contact inhibition of proliferation), so a colony settles to a homeostatic
density with growth confined to its uncrowded rim.

This demo turns that feedback on (`enable_pressure_inhibition`) and compares,
under identical well-fed conditions (nutrient is NOT limiting here -- the only
brake is mechanical), a colony grown with vs. without the stress->proliferation
gate. With it on, the compressed core arrests while the rim keeps dividing, and
the cell count plateaus.

Outputs:
  mechanics_pressure_inhibition.png  (free vs inhibited colonies coloured by
                                      contact pressure, + count-vs-time)

Run:  python experiments/mechanics_pressure_inhibition.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm, colors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.kernels.neighbors import (build_cell_list_numba,       # noqa: E402
                                        contact_pressure_celllist_numba)

HERE = os.path.dirname(os.path.abspath(__file__))
L = 140.0
STEPS = 240
THRESH = 1.5


def config(inhibit):
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 20,
        'initial_cluster_radius': 10.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': 120,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 95.0,
        'nutrient_D': 2.0, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 4.0,
        'adhesion_strength': 0.3, 'adhesion_cutoff_factor': 1.3,
        'repulsion_strength': 30.0, 'overlap_iterations': 4,
        'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 7,
        'enable_pressure_inhibition': inhibit, 'pressure_threshold': THRESH,
    }


def run(inhibit):
    sim = CellSimulation(config(inhibit), config_name='pi')
    t, n = [], []
    for step in range(STEPS):
        sim._simulation_step()
        if step % 5 == 0:
            t.append(step); n.append(len(sim.cells))
    return sim, t, n


def assign_pressure(sim):
    """Compute & store contact pressure for the current config (the free run does
    not compute it during stepping)."""
    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])
    bin_size = 2.0 * rad.max() * max(sim.adhesion_cutoff_factor, 1.0)
    order, start, nbx = build_cell_list_numba(pos, sim.physical_size, bin_size)
    p = contact_pressure_celllist_numba(pos, rad, sim.repulsion_strength,
                                        order, start, nbx, bin_size)
    for c, pi in zip(sim.cells, p):
        c.pressure = pi


def draw(ax, sim, title, pmax):
    norm = colors.Normalize(vmin=0.0, vmax=pmax)
    cmap = matplotlib.colormaps['inferno']
    for c in sim.cells:
        ax.add_patch(Circle(c.position, c.radius, color=cmap(norm(c.pressure)), lw=0))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{title}\n{len(sim.cells)} cells')
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def main():
    print("running free (no mechanical feedback) ...", flush=True)
    free, tf, nf = run(False)
    print(f"  free: {len(free.cells)} cells", flush=True)
    print("running pressure-inhibited ...", flush=True)
    inh, ti, ni = run(True)
    print(f"  inhibited: {len(inh.cells)} cells", flush=True)

    # pressure is only computed when the feature is on; compute it for the free
    # colony too for a fair coloured comparison.
    assign_pressure(free)
    pmax = max(max((c.pressure for c in free.cells), default=1.0),
               max((c.pressure for c in inh.cells), default=1.0), THRESH)

    fig = plt.figure(figsize=(15, 5.5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    draw(ax1, free, 'free growth (no feedback)', pmax)
    sm = draw(ax2, inh, f'pressure-inhibited (thr={THRESH})', pmax)
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, fraction=0.046, label='contact pressure')

    ax3.plot(tf, nf, 'o-', color='crimson', label='free')
    ax3.plot(ti, ni, 's-', color='steelblue', label='pressure-inhibited')
    ax3.axhline(ni[-1], color='steelblue', ls=':', alpha=0.5)
    ax3.set(xlabel='step', ylabel='cell count',
            title='Mechanical feedback caps proliferation (homeostasis)')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    fig.suptitle('Contact inhibition / homeostatic pressure: '
                 'free overgrows; feedback arrests the compressed core', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, 'mechanics_pressure_inhibition.png')
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"free {nf[-1]} cells vs inhibited {ni[-1]} cells "
          f"({100*(1-ni[-1]/nf[-1]):.0f}% fewer)")
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
