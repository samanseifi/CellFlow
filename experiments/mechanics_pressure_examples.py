"""Worked examples of contact-pressure (homeostatic) growth control.

Two scenarios beyond the basic free-vs-inhibited comparison:

  A. Threshold sweep -- the homeostatic set-point is tunable: a lower
     pressure_threshold arrests the colony at a lower density.

  B. Ablation -> regrowth -- a confluent, growth-arrested colony is wounded
     (centre cells removed); the cells bordering the wound lose their
     confinement, their pressure drops below threshold, and they resume
     dividing until the wound refills and the colony re-arrests. This is the
     defining behaviour of contact inhibition of proliferation (homeostatic
     tissue repair).

Outputs:
  mechanics_pressure_threshold_sweep.png
  mechanics_pressure_ablation.png

Run:  python experiments/mechanics_pressure_examples.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.kernels.neighbors import (build_cell_list_numba,       # noqa: E402
                                        contact_pressure_celllist_numba)

HERE = os.path.dirname(os.path.abspath(__file__))
L = 130.0


def config(threshold, inhibit=True):
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 18,
        'initial_cluster_radius': 9.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': 110,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 95.0,
        'nutrient_D': 2.0, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 4.0,
        'adhesion_strength': 0.3, 'adhesion_cutoff_factor': 1.3,
        'repulsion_strength': 30.0, 'overlap_iterations': 4,
        'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 7,
        'enable_pressure_inhibition': inhibit, 'pressure_threshold': threshold,
    }


def assign_pressure(sim):
    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])
    bs = 2.0 * rad.max() * max(sim.adhesion_cutoff_factor, 1.0)
    order, start, nbx = build_cell_list_numba(pos, sim.physical_size, bs)
    p = contact_pressure_celllist_numba(pos, rad, sim.repulsion_strength,
                                        order, start, nbx, bs)
    for c, pi in zip(sim.cells, p):
        c.pressure = pi


def draw(ax, sim, title, pmax):
    norm = colors.Normalize(vmin=0.0, vmax=pmax)
    cmap = matplotlib.colormaps['inferno']
    for c in sim.cells:
        ax.add_patch(Circle(c.position, c.radius, color=cmap(norm(c.pressure)), lw=0))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)


# ---------------- A. threshold sweep ----------------
def threshold_sweep():
    print("A) threshold sweep ...", flush=True)
    thresholds = [0.8, 1.5, 3.0]
    sims, counts = [], []
    for thr in thresholds:
        sim = CellSimulation(config(thr), config_name=f'thr{thr}')
        for _ in range(200):
            sim._simulation_step()
        sims.append(sim); counts.append(len(sim.cells))
        print(f"   threshold={thr}: {len(sim.cells)} cells", flush=True)
    # an uninhibited reference
    free = CellSimulation(config(1.0, inhibit=False), config_name='free')
    for _ in range(200):
        free._simulation_step()
    assign_pressure(free)
    print(f"   free (off): {len(free.cells)} cells", flush=True)

    pmax = max(max((c.pressure for c in s.cells), default=1) for s in sims + [free])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for a, s, thr in zip(ax[:3], sims, thresholds):
        draw(a, s, f'threshold={thr}\n{len(s.cells)} cells', pmax)
    draw(ax[3], free, f'no inhibition\n{len(free.cells)} cells', pmax)
    fig.suptitle('Homeostatic set-point is tunable: lower pressure threshold -> '
                 'lower arrested density', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, 'mechanics_pressure_threshold_sweep.png')
    fig.savefig(out, dpi=115); plt.close(fig)
    print(f"   -> {out}")


# ---------------- B. ablation -> regrowth ----------------
def _grow_to_plateau(sim, t, n, step0, max_extra, win=12, min_steps=120):
    """Step until the cell count is flat (confluent/arrested) or max_extra hit."""
    step = step0
    end = step0 + max_extra
    while step < end:
        sim._simulation_step(); step += 1
        if step % 5 == 0:
            t.append(step); n.append(len(sim.cells))
            if (step - step0) > min_steps and len(n) > win and \
               max(n[-win:]) - min(n[-win:]) <= 2:     # count stable -> plateau
                break
    return step


def ablation():
    print("B) ablation -> regrowth (confined -> true count homeostasis) ...", flush=True)
    LB = 64.0                                   # small box: colony fills it (cells
    cfg = config(1.5)                           # are clamped to the domain), so the
    cfg.update(physical_size=LB, grid_resolution=64,   # total count -- not just the
              num_cells=10, initial_cluster_radius=6.0)  # density -- reaches a plateau
    sim = CellSimulation(cfg, config_name='ablate')
    center = np.array([LB / 2, LB / 2])
    t, n = [], []
    snaps = {}
    ABLATE_R = 16.0

    step = _grow_to_plateau(sim, t, n, 0, 900)            # grow to confluent arrest
    assign_pressure(sim)
    snaps['before'] = [(c.position.copy(), c.radius, c.pressure) for c in sim.cells]
    wound_step = step
    sim.cells = [c for c in sim.cells
                 if np.linalg.norm(c.position - center) > ABLATE_R]   # wound
    snaps['after'] = [(c.position.copy(), c.radius, 0.0) for c in sim.cells]
    t.append(step); n.append(len(sim.cells))
    _grow_to_plateau(sim, t, n, step, 900)               # heal -> re-arrest
    assign_pressure(sim)
    snaps['healed'] = [(c.position.copy(), c.radius, c.pressure) for c in sim.cells]
    print(f"   before wound: {len(snaps['before'])}, after: {len(snaps['after'])}, "
          f"healed: {len(snaps['healed'])} (wound at step {wound_step})", flush=True)

    pmax = max((p for _, _, p in snaps['before'] + snaps['healed']), default=1.0)
    norm = colors.Normalize(vmin=0.0, vmax=pmax)
    cmap = matplotlib.colormaps['inferno']
    fig = plt.figure(figsize=(16, 4.8))
    titles = [('before', 'arrested (confluent)'), ('after', 'wounded (centre removed)'),
              ('healed', 'regrown & re-arrested')]
    for k, (key, ttl) in enumerate(titles):
        a = fig.add_subplot(1, 4, k + 1)
        for p, r, pr in snaps[key]:
            a.add_patch(Circle(p, r, color=cmap(norm(pr)), lw=0))
        a.set_xlim(0, LB); a.set_ylim(0, LB); a.set_aspect('equal')
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f'{ttl}\n{len(snaps[key])} cells', fontsize=10)
    axc = fig.add_subplot(1, 4, 4)
    axc.plot(t, n, 'o-', color='seagreen')
    axc.axvline(wound_step, color='crimson', ls='--', alpha=0.7, label='wound')
    axc.set(xlabel='step', ylabel='cell count',
            title='count: arrest -> wound -> regrow -> re-arrest')
    axc.legend(); axc.grid(True, alpha=0.3)
    fig.suptitle('Contact inhibition releases at a wound: proliferation resumes '
                 'where pressure drops, then re-arrests (homeostatic repair)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(HERE, 'mechanics_pressure_ablation.png')
    fig.savefig(out, dpi=115); plt.close(fig)
    print(f"   -> {out}")


def main():
    threshold_sweep()
    ablation()
    print("Done.")


if __name__ == '__main__':
    main()
