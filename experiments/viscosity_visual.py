"""Visual snapshots of the cell cluster vs viscosity (Brinkman/IBM model).

Renders a grid of snapshots: rows = viscosity, columns = simulation step.
Each panel shows the nutrient field (background), cells (circles), and the
fluid velocity field (quiver, normalized to show the flow pattern since its
magnitude scales as 1/mu). Demonstrates that low viscosity lets the cluster
move/relax while high viscosity nearly freezes it.

Produces experiments/viscosity_visual.png.
Run:  python experiments/viscosity_visual.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402
from experiments.viscosity_sweep import base_config  # noqa: E402


SNAP_STEPS = [0, 40, 120]
MUS = [1.0, 64.0]


def capture(mu, seed=12345):
    np.random.seed(seed)
    sim = CellSimulation(base_config(mu), config_name=f'vis_mu{mu:g}')
    snaps = {}
    last = max(SNAP_STEPS)
    for step in range(last + 1):
        if step in SNAP_STEPS:
            snaps[step] = {
                'pos': np.array([c.position for c in sim.cells]),
                'rad': np.array([c.radius for c in sim.cells]),
                'phase': [c.phase for c in sim.cells],
                'u': sim.fluid_velocity.copy(),
                'nutrient': sim.nutrient_field.copy(),
                'vmax': float(np.max(np.linalg.norm(sim.fluid_velocity, axis=2))),
            }
        if step < last:
            sim._simulation_step()
    return sim, snaps


def draw_panel(ax, snap, L, title, init_rg=None, com=None):
    ax.set_facecolor('#0b1021')
    # faint nutrient background
    ax.imshow(snap['nutrient'], cmap='viridis', origin='lower',
              extent=[0, L, 0, L], alpha=0.35)

    # initial cluster extent (dashed) to make spreading visible
    if init_rg is not None and com is not None:
        ax.add_artist(plt.Circle(com, init_rg, color='cyan', fill=False,
                                 lw=1.2, ls='--', alpha=0.7))

    for p, r, ph in zip(snap['pos'], snap['rad'], snap['phase']):
        color = 'red' if ph == 'DIVISION' else 'deepskyblue'
        ax.add_artist(plt.Circle(p, r, color=color, alpha=0.95))
        ax.add_artist(plt.Circle(p, r, color='white', fill=False, lw=0.6))

    # normalized velocity quiver (pattern, since magnitude ~ 1/mu)
    u = snap['u']
    n = u.shape[0]
    s = max(1, n // 22)
    xs = np.arange(0, n, s) * (L / n)
    X, Y = np.meshgrid(xs, xs)
    ux = u[::s, ::s, 0]
    uy = u[::s, ::s, 1]
    mag = np.hypot(ux, uy)
    mag[mag == 0] = 1.0
    ax.quiver(X, Y, ux / mag, uy / mag, color='orange', alpha=0.55,
              scale=38, width=0.004, headwidth=3)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    L = base_config(1.0)['physical_size']
    fig, axes = plt.subplots(len(MUS), len(SNAP_STEPS),
                             figsize=(4 * len(SNAP_STEPS), 4 * len(MUS)))
    for row, mu in enumerate(MUS):
        sim, snaps = capture(mu)
        s0 = snaps[SNAP_STEPS[0]]
        com0 = s0['pos'].mean(axis=0)
        init_rg = float(np.sqrt(np.mean(np.sum((s0['pos'] - com0) ** 2, axis=1))))
        for col, step in enumerate(SNAP_STEPS):
            sn = snaps[step]
            draw_panel(axes[row, col], sn, L,
                       f"mu={mu:.0f}, step={step}\n(max |u|={sn['vmax']:.3g}, "
                       f"N={len(sn['pos'])})",
                       init_rg=init_rg, com=com0)
    fig.suptitle("Cluster vs viscosity — Brinkman/IBM fluid "
                 "(arrows = flow direction, normalized)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viscosity_visual.png')
    fig.savefig(out, dpi=110)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
