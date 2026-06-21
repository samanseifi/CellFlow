"""Diffusion-limited cell aggregate: the necrotic-core / critical-size problem.

The central constraint on growing cell aggregates (spheroids, microcarrier
clumps) in a bioreactor: nutrient/oxygen reaches the interior only by diffusion
from the surrounding medium, against cellular uptake.  Past a critical size the
core can no longer be supplied, goes anoxic, and becomes quiescent/necrotic --
only a thin VIABLE SHELL at the rim survives.  The shell thickness (the diffusion
penetration depth) sets the maximum useful aggregate size and is governed by the
balance of medium supply, diffusivity, and metabolic uptake.

This demo reproduces that with CellFlow's nutrient field + uptake kernel +
active/passive (quiescence) transition, in the fast-diffusion / quasi-steady
limit: a FIXED aggregate is bathed in fresh medium and we solve diffusion +
first-order uptake to steady state.  (Growth would advance the rim, but the
steady viable-shell thickness -- the design quantity -- is set by transport, not
kinetics, so a frozen aggregate isolates it cleanly and robustly.)

  Part A (visual):  one aggregate, fresh medium at t=0; the core depletes and
                    goes necrotic (dark) while a viable green shell remains.
                    -> bioreactor_aggregate.gif, _montage.png, _timeseries.png

  Part B (design):  viable fraction & shell vs aggregate size, at two supply
                    levels -> the critical-size curve (bigger aggregates necrose;
                    richer medium pushes the critical size up).
                    -> bioreactor_critical_size.png

Run:  python experiments/bioreactor_aggregate.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.kernels.diffusion import diffuse_field_numba           # noqa: E402
from cellflow.kernels.fields import absorb_nutrient_numba            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
L = 320.0
G = 320
DX = L / G
DT = 0.05
NUTRIENT_D = 4.0          # medium diffusivity (fast -> quasi-steady)
CONSUMPTION = 1.0         # per-cell metabolic uptake (tuned: clear core by ~800 steps)
THRESH = 15.0             # local supply below this -> quiescent / necrotic
CELL_R = 2.4
VIABLE = 'limegreen'
NECROTIC = '#4a1414'


def hex_disk(R):
    """Frozen hex-packed disk of cell centres (radius R) centred in the box."""
    spacing = 1.95 * CELL_R
    c = L / 2
    pts = []
    n = int(2 * R / (spacing * np.sqrt(3) / 2)) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-int(2 * R / spacing) - 2, int(2 * R / spacing) + 2):
            x = i * spacing + xoff
            if np.hypot(x, y) <= R:
                pts.append((c + x, c + y))
    return np.array(pts)


def equilibrate(positions, bc, steps, snap_every=0):
    """Diffuse + uptake to (quasi-)steady. Returns field, history, frame list."""
    field = np.full((G, G), float(bc))
    c = np.array([L / 2, L / 2])
    dist = np.array([np.hypot(p[0] - c[0], p[1] - c[1]) for p in positions])
    hist = {'t': [], 'viable': []}
    frames = []
    for step in range(steps):
        field = diffuse_field_numba(field, NUTRIENT_D, DT, DX, float(bc))
        read = field.copy()
        for p in positions:
            absorb_nutrient_numba(np.array(p), CELL_R, field, read, DT, CONSUMPTION, DX)
        if step % 5 == 0:
            vals = sample(field, positions)
            hist['t'].append(step); hist['viable'].append((vals > THRESH).mean())
        if snap_every and step % snap_every == 0:
            frames.append(frame(field, positions, dist, bc, step))
    return field, hist, frames


def sample(field, positions):
    return np.array([field[int(p[1] / DX), int(p[0] / DX)] for p in positions])


def shell_thickness(field, positions):
    """Viable shell = outer radius - innermost viable radius."""
    c = np.array([L / 2, L / 2])
    dist = np.array([np.hypot(p[0] - c[0], p[1] - c[1]) for p in positions])
    viable = sample(field, positions) > THRESH
    r_out = dist.max()
    return (r_out - dist[viable].min()) if viable.any() else 0.0


def frame(field, positions, dist, bc, step):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(field, cmap='magma', origin='lower', extent=[0, L, 0, L],
              vmin=0, vmax=bc)
    vals = sample(field, positions)
    for p, v in zip(positions, vals):
        ax.add_patch(Circle(p, CELL_R, color=VIABLE if v > THRESH else NECROTIC, lw=0))
    via = 100 * (vals > THRESH).mean()
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'step {step}    viable {via:.0f}%')
    fig.tight_layout()
    path = os.path.join(HERE, f'_bf_{step:04d}.png')
    fig.savefig(path, dpi=85); plt.close(fig)
    return path


def main():
    # -------- Part A: one aggregate, necrotic core emerges --------
    print("Part A: one aggregate in fresh medium -> necrotic core ...", flush=True)
    R_DEMO, BC_DEMO, STEPS, SNAP = 90.0, 70.0, 900, 30
    pos = hex_disk(R_DEMO)
    print(f"  aggregate R={R_DEMO:.0f}  ({len(pos)} cells)", flush=True)
    field, hist, frames = equilibrate(pos, BC_DEMO, STEPS, SNAP)

    gif = os.path.join(HERE, 'bioreactor_aggregate.gif')
    with imageio.get_writer(gif, mode='I', duration=0.1) as w:
        for f in frames:
            w.append_data(imageio.imread(f))
    print(f"  -> {gif} ({len(frames)} frames, "
          f"final viable {100*hist['viable'][-1]:.0f}%)", flush=True)

    pick = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, 5)]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.3))
    for a, f in zip(ax, pick):
        a.imshow(imageio.imread(f)); a.axis('off')
    fig.suptitle('Diffusion-limited aggregate: fresh medium -> anoxic necrotic core, '
                 'viable green shell', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(HERE, 'bioreactor_aggregate_montage.png'), dpi=110)
    plt.close(fig)
    for f in frames:
        os.remove(f)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(hist['t'], 100 * np.array(hist['viable']), 'o-', color='firebrick')
    ax.set(xlabel='diffusion step', ylabel='viable fraction (%)',
           title=f'Necrotic core develops as the medium is consumed (R={R_DEMO:.0f})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'bioreactor_aggregate_timeseries.png'), dpi=110)
    plt.close(fig)

    # -------- Part B: critical-size curve --------
    print("Part B: viable fraction vs aggregate size, two supply levels ...", flush=True)
    radii = [25.0, 50.0, 75.0, 100.0]
    fig, (axf, axs) = plt.subplots(1, 2, figsize=(14, 5))
    for bc, col in [(40.0, 'steelblue'), (120.0, 'darkorange')]:
        fracs, shells = [], []
        for R in radii:
            p = hex_disk(R)
            f, _, _ = equilibrate(p, bc, 800)
            v = sample(f, p)
            fracs.append(100 * (v > THRESH).mean())
            shells.append(shell_thickness(f, p))
            print(f"  supply={bc:5.0f} R={R:5.0f}: viable {fracs[-1]:4.0f}%  "
                  f"shell {shells[-1]:5.1f}u", flush=True)
        axf.plot(radii, fracs, 'o-', color=col, lw=2, label=f'supply={bc:.0f}')
        axs.plot(radii, shells, 's-', color=col, lw=2, label=f'supply={bc:.0f}')
    axf.axhline(50, color='k', ls=':', alpha=0.5)
    axf.set(xlabel='aggregate radius (units)', ylabel='viable fraction (%)',
            title='Larger aggregates necrose; richer medium raises critical size')
    axf.legend(); axf.grid(True, alpha=0.3)
    axs.set(xlabel='aggregate radius (units)', ylabel='viable shell thickness (units)',
            title='Viable shell ~ constant (transport-set), independent of size')
    axs.legend(); axs.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, 'bioreactor_critical_size.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  -> {out}\nDone.")


if __name__ == '__main__':
    main()
