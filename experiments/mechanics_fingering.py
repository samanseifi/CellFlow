"""Does mechanical feedback (pressure-inhibited growth) destabilize a colony front?

Hypothesis (cf. the Giverso branching study): pressure inhibition arrests the
crowded interior so only the rim proliferates, AND a convex protrusion tip has
FEWER neighbours -> lower contact pressure -> keeps dividing, while a concave
valley is more crowded -> arrests. That curvature feedback should AMPLIFY front
perturbations (fingering). Caveat: contact pressure is short-range and fights the
steric surface tension, so pressure alone may only roughen the front; combining
it with nutrient limitation (long-range Mullins-Sekerka tip focusing) may be
stronger.

We seed a mode-m lobed colony and track the seeded-mode amplitude and front
roughness under four regimes:
  control        : well fed, no feedback                 (expected: heals -> round)
  pressure       : well fed + pressure inhibition         (the test)
  nutrient       : diffusion-limited rim (quiescence)     (the Giverso regime)
  both           : diffusion-limited + pressure inhibition

Output: mechanics_fingering.png  (final fronts + amplitude/roughness vs time)

Run:  python experiments/mechanics_fingering.py
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

L = 200.0
G = 160
R0 = 50.0
MODE = 6
EPS = 0.18
STEPS = 240
CELL_R = 2.4


def base_config():
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 1,
        'initial_cluster_radius': 1.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 90.0,
        'nutrient_D': 6.0, 'chi_nutrient': 0.0, 'diffusion_solver': 'implicit',
        'walk_speed': 0.0, 'max_propulsive_force': 2.0,
        'adhesion_strength': 0.0, 'adhesion_cutoff_factor': 1.2,
        'repulsion_strength': 30.0, 'overlap_iterations': 4,
        'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 600.0, 'fluid_model': 'brinkman_fft',
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 5,
    }


REGIMES = {
    'control':  dict(cons=0.15, quiesce=False, pressure=False),
    'pressure': dict(cons=0.15, quiesce=False, pressure=True),
    'nutrient': dict(cons=0.8,  quiesce=True,  pressure=False),
    'both':     dict(cons=0.8,  quiesce=True,  pressure=True),
}
COLORS = {'control': 'gray', 'pressure': 'crimson',
          'nutrient': 'steelblue', 'both': 'seagreen'}


def lobed_colony(cons, cell_r=CELL_R):
    spacing = 1.9 * cell_r
    Cell.next_id = 0
    cells = []
    c = L / 2
    n = int(2 * R0 * (1 + EPS) / (spacing * np.sqrt(3) / 2)) + 2
    nx = int(2 * R0 * (1 + EPS) / spacing) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * spacing + xoff
            r = np.hypot(x, y); th = np.arctan2(y, x)
            if r <= R0 * (1 + EPS * np.cos(MODE * th)):
                cell = Cell(np.array([c + x, c + y]), nutrient=55.0, area_conserving=True)
                cell.radius = cell_r
                cell.consumption_rate = cons
                cells.append(cell)
    return cells


def config_for(reg):
    cfg = base_config()
    if reg['quiesce']:
        cfg['enable_quiescence'] = True
        cfg['quiescence_nutrient_threshold'] = 25.0
    if reg['pressure']:
        cfg['enable_pressure_inhibition'] = True
        cfg['pressure_threshold'] = 1.5
    return cfg


def front_radii(cells, center, n_bins=240):
    pos = np.array([c.position for c in cells])
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


def mode_amp(R, m):
    f = np.fft.rfft(R - R.mean())
    return 2.0 * np.abs(f[m]) / len(R)


def run(name, reg):
    cfg = config_for(reg)
    sim = CellSimulation(cfg, config_name=f'fing_{name}')
    sim.cells = lobed_colony(reg['cons'])
    # enforce metabolism each step (daughters inherit consumption via divide; set
    # explicitly to be safe)
    center = np.array([L / 2, L / 2])
    R = front_radii(sim.cells, center)
    a0 = mode_amp(R, MODE) / R.mean()
    r0 = R.std() / R.mean()
    t, amp, rough = [0], [1.0], [1.0]
    for step in range(STEPS):
        for c in sim.cells:
            c.consumption_rate = reg['cons']
        sim._simulation_step()
        if step % 10 == 0:
            R = front_radii(sim.cells, center)
            t.append(step + 1)
            amp.append((mode_amp(R, MODE) / R.mean()) / a0)
            rough.append((R.std() / R.mean()) / r0)
    print(f"  {name:9s}: {len(sim.cells):5d} cells, "
          f"mode-{MODE} {amp[-1]:.2f}x, roughness {rough[-1]:.2f}x", flush=True)
    return sim, t, amp, rough


def main():
    results = {}
    for name, reg in REGIMES.items():
        print(f"running {name} ...", flush=True)
        results[name] = run(name, reg)

    fig = plt.figure(figsize=(18, 9))
    for k, name in enumerate(REGIMES):
        sim = results[name][0]
        ax = fig.add_subplot(2, 4, k + 1)
        active_attr = any(getattr(c, 'active', True) is False for c in sim.cells)
        for c in sim.cells:
            col = ('limegreen' if getattr(c, 'active', True) else '#21303a')
            ax.add_patch(Circle(c.position, c.radius, color=col, lw=0))
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{name}\n{len(sim.cells)} cells', fontsize=11)

    axA = fig.add_subplot(2, 2, 3)
    axR = fig.add_subplot(2, 2, 4)
    for name in REGIMES:
        _, t, amp, rough = results[name]
        axA.plot(t, amp, 'o-', color=COLORS[name], label=name, ms=4)
        axR.plot(t, rough, 's-', color=COLORS[name], label=name, ms=4)
    for ax, ttl, yl in [(axA, f'seeded mode-{MODE} amplitude', 'amp / initial'),
                        (axR, 'front roughness', 'roughness / initial')]:
        ax.axhline(1.0, color='k', ls=':', alpha=0.5)
        ax.set(xlabel='step', ylabel=yl, title=ttl)
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('Does pressure-inhibited growth finger a colony front? '
                 '(seeded mode-6; >1 = amplifying = unstable)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mechanics_fingering.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
