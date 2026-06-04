"""Cluster growth -> animated GIF, with cleaner division packing and
area-conserving ELLIPSE shapes from contact stress.

A small seed grows in a Dirichlet nutrient bath. Two fidelity touches:
  - daughters are born just touching the parent (issue #21 fix) and overlaps
    are relaxed with a few sweeps per step, so cells don't intertwine;
  - each cell is DRAWN as an area-conserving ellipse whose elongation comes
    from the local contact stress (issue #22) -- rim cells stretch tangentially,
    interior cells stay round. Mechanics remain circular (cheap).

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

L = 100.0
G = 100
STEPS = 400
SAVE_EVERY = 8
MONTAGE_STEPS = [0, 90, 180, 280, STEPS - 1]
AR_GAIN = 1.6        # how strongly contact anisotropy elongates a cell
AR_MAX = 2.2         # cap on aspect ratio


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
        'overlap_iterations': 4,
    }


def cell_shapes(positions, radii):
    """Area-conserving ellipse (semi-axes a>=b, angle) from contact stress.

    Build M = sum_j w_ij (u_ij outer u_ij), w_ij = overlap-ish weight, u_ij the
    unit vector to neighbour j. The cell is compressed most along M's top
    eigenvector, so it elongates along the OTHER axis (area conserved).
    """
    n = len(positions)
    if n == 0:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
    diff = positions[None, :, :] - positions[:, None, :]          # (n,n,2): j - i
    dist = np.sqrt((diff ** 2).sum(-1)) + np.eye(n) * 1e9          # avoid self
    touch = radii[:, None] + radii[None, :]
    w = np.maximum(0.0, touch * 1.2 - dist)                       # contact weight
    ux, uy = diff[..., 0] / dist, diff[..., 1] / dist
    Mxx = (w * ux * ux).sum(1)
    Myy = (w * uy * uy).sum(1)
    Mxy = (w * ux * uy).sum(1)
    # 2x2 symmetric eigenvalues
    tr = Mxx + Myy
    det_term = np.sqrt(np.maximum(0.0, ((Mxx - Myy) / 2) ** 2 + Mxy ** 2))
    lam1 = tr / 2 + det_term
    lam2 = tr / 2 - det_term
    aniso = (lam1 - lam2) / (lam1 + lam2 + 1e-9)
    AR = np.minimum(AR_MAX, 1.0 + AR_GAIN * aniso)
    # compression axis angle (top eigenvector); elongate perpendicular (+90 deg)
    comp_angle = 0.5 * np.arctan2(2 * Mxy, Mxx - Myy)
    elong_angle = comp_angle + np.pi / 2
    a = radii * np.sqrt(AR)        # along elongation
    b = radii / np.sqrt(AR)        # area = pi a b = pi r^2 conserved
    return a, b, elong_angle


def draw_cells(ax, positions, radii, phases):
    a, b, ang = cell_shapes(positions, radii)
    for k in range(len(positions)):
        color = 'red' if phases[k] == 'DIVISION' else 'white'
        ax.add_patch(Ellipse(positions[k], 2 * a[k], 2 * b[k],
                             angle=np.degrees(ang[k]), color=color, alpha=0.85,
                             ec='black', lw=0.4))


def render_frame(nutrient, positions, radii, phases, step, outdir):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(nutrient, cmap='viridis', origin='lower', extent=[0, L, 0, L])
    draw_cells(ax, positions, radii, phases)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_title(f'step {step}   cells={len(positions)}')
    fp = os.path.join(outdir, f'frame_{step:04d}.png')
    fig.savefig(fp, dpi=90)
    plt.close(fig)
    return fp


def main():
    import imageio
    sim = CellSimulation(config(), config_name='cluster_growth')
    frames, snaps, counts = [], {}, []
    for step in range(STEPS):
        sim._simulation_step()
        counts.append(len(sim.cells))
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        ph = [c.phase for c in sim.cells]
        if step % SAVE_EVERY == 0 or step == STEPS - 1:
            frames.append(render_frame(sim.nutrient_field, pos, rad, ph, step, sim.output_dir))
        if step in MONTAGE_STEPS:
            snaps[step] = (sim.nutrient_field.copy(), pos, rad, ph)
        print(f"\rstep {step+1}/{STEPS}  cells={len(sim.cells)}", end="")
    print()

    fig, axes = plt.subplots(1, len(MONTAGE_STEPS), figsize=(4 * len(MONTAGE_STEPS), 4.2))
    for ax, step in zip(axes, MONTAGE_STEPS):
        nut, pos, rad, ph = snaps[step]
        ax.imshow(nut, cmap='viridis', origin='lower', extent=[0, L, 0, L])
        draw_cells(ax, pos, rad, ph)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}   cells={len(pos)}", fontsize=10)
    fig.suptitle("Cluster growth with ellipse cell shapes (contact-stress deformation)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    montage = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'cluster_growth_montage.png')
    fig.savefig(montage, dpi=110)
    print(f"cells: {counts[0]} -> {counts[-1]}")
    print(f"Saved montage -> {montage}")

    with imageio.get_writer('cluster_growth_simulation.gif', mode='I', duration=0.1) as w:
        for fp in frames:
            w.append_data(imageio.imread(fp))
            os.remove(fp)
    print("Saved GIF -> cluster_growth_simulation.gif")


if __name__ == '__main__':
    main()
