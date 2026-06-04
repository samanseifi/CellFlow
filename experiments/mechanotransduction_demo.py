"""Visualize mechanotransduction: cell polarity aligning to fluid shear (#17).

A field of cells sits in an imposed steady simple-shear flow u = (gamma*y, 0).
Each cell senses the local strain rate and rotates its polarity (a nematic
director) toward the principal strain axis (+45 deg) at a rate set by the shear.
We show director snapshots (disordered -> aligned) and the nematic order
parameter vs time.

Produces experiments/mechanotransduction_demo.png (and optionally a GIF).
Run:  python experiments/mechanotransduction_demo.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402

L = 60.0
G = 120
DX = L / G
GAMMA = 1.0
SNAPS = [0, 15, 120]
ITERS = max(SNAPS)


def build():
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 1, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G, 'nutrient_bc_value': 20.0,
        'nutrient_D': 0.5, 'chi_nutrient': 0.0, 'walk_speed': 0.0,
        'max_propulsive_force': 0.0, 'viscosity': 1.0, 'adhesion_strength': 0.0,
        'adhesion_cutoff_factor': 1.5, 'repulsion_strength': 0.0,
        'attractant_D': 0.0, 'chi_attractant': 0.0, 'enable_visualization': False,
        'seed': 0, 'enable_mechanotransduction': True, 'shear_alignment_rate': 2.0,
    }
    sim = CellSimulation(cfg, config_name='mechdemo')
    # lattice of cells, away from the periodic y-wrap rows
    xs = np.arange(10, 51, 5.0)
    ys = np.arange(16, 45, 5.0)
    from cellflow.cell import Cell
    Cell.next_id = 0
    sim.cells = []
    for x in xs:
        for y in ys:
            c = Cell(np.array([x, y]))
            sim.cells.append(c)
    # imposed simple-shear velocity field (u_x = gamma * y)
    sim.fluid_velocity = np.zeros((G, G, 2))
    sim.fluid_velocity[:, :, 0] = GAMMA * (np.arange(G) * DX)[:, None]
    return sim


def order_parameter(sim):
    """Nematic order S = |<exp(2 i phi)>|; mean axis angle."""
    phi = np.array([c.polarity for c in sim.cells])
    c2, s2 = np.mean(np.cos(2 * phi)), np.mean(np.sin(2 * phi))
    return np.hypot(c2, s2), 0.5 * np.arctan2(s2, c2)


def draw_directors(ax, sim, title):
    for c in sim.cells:
        x, y = c.position
        d = 1.8
        dxp, dyp = d * np.cos(c.polarity), d * np.sin(c.polarity)
        # color by closeness to the 45-deg strain axis
        align = abs(np.cos(2 * (c.polarity - np.pi / 4)))
        ax.plot([x - dxp, x + dxp], [y - dyp, y + dyp], '-',
                color=plt.cm.viridis(align), lw=2)
    ax.set_xlim(5, 55); ax.set_ylim(12, 48); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    sim = build()
    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])

    snaps = {}
    S_series, ang_series = [], []
    for it in range(ITERS + 1):
        if it in SNAPS:
            snaps[it] = [c.polarity for c in sim.cells]
        S, ang = order_parameter(sim)
        S_series.append(S); ang_series.append(np.degrees(ang))
        if it < ITERS:
            sim._update_polarity(pos, rad)

    fig = plt.figure(figsize=(16, 4.5))
    for col, it in enumerate(SNAPS):
        ax = fig.add_subplot(1, len(SNAPS) + 1, col + 1)
        for c, p in zip(sim.cells, snaps[it]):
            c.polarity = p
        draw_directors(ax, sim, f"step {it}   (S={S_series[it]:.2f})")
    # restore final
    for c, p in zip(sim.cells, snaps[SNAPS[-1]]):
        c.polarity = p

    ax = fig.add_subplot(1, len(SNAPS) + 1, len(SNAPS) + 1)
    ax.plot(S_series, label='nematic order S')
    ax.plot(np.array(ang_series) / 45.0, label='mean angle / 45°')
    ax.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax.set(xlabel='step', ylim=(0, 1.1),
           title='Alignment to the strain axis')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Mechanotransduction: cell polarity aligns to fluid shear "
                 "(simple shear, flow → +x; strain axis = 45°)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'mechanotransduction_demo.png')
    fig.savefig(out, dpi=110)
    print(f"final nematic order S = {S_series[-1]:.3f}, "
          f"mean axis = {ang_series[-1]:.1f} deg (target 45)")
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
