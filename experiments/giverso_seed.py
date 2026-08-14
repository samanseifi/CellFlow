"""Decisive linear-stability test for colony fingering.

Instead of waiting for cell-scale noise to seed an l-scale instability (which the
scale gap forbids), we SEED a clean l-scale front modulation directly -- a
colony with a lobed boundary R(theta) = R0 (1 + eps cos(m theta)) -- and watch
whether the seeded mode m GROWS (front linearly unstable: fingering is only a
seeding problem) or DECAYS back to round (front stable: surface tension wins).
Run in the diffusion-limited regime with quiescence + gradient-directed division.

Outputs the initial vs final colony and the seeded-mode amplitude vs time.
Run:  python experiments/giverso_seed.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.cell import Cell                                      # noqa: E402

L = 600.0
G = 400
R0 = 90.0
SEED_MODE = 6          # number of lobes seeded
SEED_EPS = 0.35        # lobe amplitude (fraction of R0) -- a big, clear seed
STEPS = 700


def config():
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 1,
        'initial_cluster_radius': 1.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 40.0,
        'nutrient_D': 0.02, 'chi_nutrient': 0.0,
        'walk_speed': 0.05, 'max_propulsive_force': 2.0,
        'adhesion_strength': 0.0, 'adhesion_cutoff_factor': 1.2,
        'repulsion_strength': 35.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': 15.0, 'overlap_iterations': 1,
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 4,
        'enable_quiescence': True, 'quiescence_nutrient_threshold': 28.0,
        'directed_division': True,
    }


def lobed_colony(cell_r=2.4, spacing=None):
    if spacing is None:
        spacing = 1.9 * cell_r
    Cell.next_id = 0
    cells = []
    c = L / 2
    n = int(2 * R0 * (1 + SEED_EPS) / (spacing * np.sqrt(3) / 2)) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        nx = int(2 * R0 * (1 + SEED_EPS) / spacing) + 2
        for i in range(-nx, nx + 1):
            x = i * spacing + xoff
            r = np.hypot(x, y)
            th = np.arctan2(y, x)
            if r <= R0 * (1 + SEED_EPS * np.cos(SEED_MODE * th)):
                cell = Cell(np.array([c + x, c + y]), nutrient=60.0, area_conserving=True)
                cell.radius = cell_r
                cells.append(cell)
    return cells


def front_radii(sim, center, n_bins=180):
    pos = np.array([c.position for c in sim.cells])
    d = pos - center
    th = np.arctan2(d[:, 1], d[:, 0]); rad = np.hypot(d[:, 0], d[:, 1])
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(th, edges) - 1
    R = np.full(n_bins, np.nan)
    for b in range(n_bins):
        rr = rad[idx == b]
        if rr.size:
            R[b] = rr.max()
    good = ~np.isnan(R)
    return np.interp(np.arange(n_bins), np.where(good)[0], R[good], period=n_bins)


def mode_amplitude(R, m):
    f = np.fft.rfft(R - R.mean())
    return 2.0 * np.abs(f[m]) / len(R)


def main():
    sim = CellSimulation(config(), config_name='giverso_seed')
    sim.cells = lobed_colony()
    center = np.array([L / 2, L / 2])

    snap0 = [(c.position.copy(), c.radius, c.active) for c in sim.cells]
    amp0 = mode_amplitude(front_radii(sim, center), SEED_MODE)
    Rmean0 = front_radii(sim, center).mean()

    times, amps = [0], [amp0 / Rmean0]
    for step in range(STEPS):
        sim._simulation_step()
        if step % 10 == 0:
            R = front_radii(sim, center)
            times.append(step + 1)
            amps.append(mode_amplitude(R, SEED_MODE) / R.mean())

    amp_final = amps[-1]
    verdict = "GROWS -> front UNSTABLE (fingering is a seeding problem)" if amp_final > amps[0] \
        else "DECAYS -> front STABLE (surface tension wins)"

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (snap, ttl) in zip(ax[:2], [
            (snap0, f"seeded: mode {SEED_MODE}, eps={SEED_EPS}"),
            ([(c.position, c.radius, c.active) for c in sim.cells],
             f"after {STEPS} steps ({len(sim.cells)} cells)")]):
        for p, r, act in snap:
            a.add_patch(Circle(p, r, color='limegreen' if act else '#243030', lw=0))
        a.set_xlim(0, L); a.set_ylim(0, L); a.set_aspect('equal')
        a.set_xticks([]); a.set_yticks([]); a.set_title(ttl)
    ax[2].plot(times, np.array(amps) / amps[0], 'o-')
    ax[2].axhline(1.0, color='k', ls=':', alpha=0.6)
    ax[2].set(xlabel='step', ylabel=f'mode-{SEED_MODE} amplitude / initial',
              title='seeded-mode growth/decay')
    ax[2].grid(True, alpha=0.3)
    fig.suptitle(f"Seeded l-scale lobe test: {verdict}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'giverso_seed.png')
    fig.savefig(out, dpi=120)
    print(f"mode-{SEED_MODE} amplitude/R: {amps[0]:.4f} -> {amp_final:.4f}  ({amp_final/amps[0]:.2f}x)")
    print(verdict)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
