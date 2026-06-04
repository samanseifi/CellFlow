"""Isolate the viscosity effect: directed cluster MIGRATION in an imposed
chemical gradient (Brinkman/IBM fluid).

Why migration (center of mass) rather than spreading: the cluster's internal
spreading is dominated by the viscosity-independent overlap-resolution step,
which conserves the center of mass. The COM can only move under the net fluid
force, whose induced velocity scales as 1/mu. So COM displacement is a clean,
overlap-resolution-proof probe of viscosity.

Setup: a fixed horizontal nutrient ramp (re-imposed each step) drives constant
chemotaxis in +x; walk_speed=0 makes it deterministic. Low viscosity migrates
far, high viscosity barely moves.

Produces experiments/viscosity_migration.png and a COM-displacement table.
Run:  python experiments/viscosity_migration.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402


L = 80.0
G = 80
MUS = [1.0, 8.0, 64.0]
STEPS = 100


def config(mu):
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 50,
        'initial_cluster_radius': 7.0, 'dt': 0.01,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_value': 20.0, 'nutrient_D': 0.5, 'chi_nutrient': 10.0,
        'walk_speed': 0.0, 'max_propulsive_force': 30.0, 'viscosity': mu,
        'adhesion_strength': 0.3, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': 20.0,
    }


def ramp_field():
    """Nutrient increasing along +x -> chemotactic drive points +x."""
    row = np.linspace(5.0, 40.0, G)
    return np.tile(row, (G, 1))


def run(mu, seed=12345):
    np.random.seed(seed)
    sim = CellSimulation(config(mu), config_name=f'mig_mu{mu:g}')
    ramp = ramp_field()
    com = []
    start_pos = np.array([c.position for c in sim.cells]).copy()
    for _ in range(STEPS):
        sim.nutrient_field = ramp.copy()        # hold the external gradient fixed
        sim._simulation_step()
        p = np.array([c.position for c in sim.cells])
        com.append(p.mean(axis=0))
    end_pos = np.array([c.position for c in sim.cells])
    com = np.array(com)
    return dict(mu=mu, start=start_pos, end=end_pos, com=com,
                dx=com[-1, 0] - start_pos[:, 0].mean())


def main():
    results = [run(mu) for mu in MUS]

    print(f"{'mu':>6} {'COM +x shift':>14} {'shift*mu':>10}")
    for r in results:
        print(f"{r['mu']:>6.0f} {r['dx']:>14.4f} {r['dx']*r['mu']:>10.3f}")
    prod = [r['dx'] * r['mu'] for r in results]
    print(f"\nshift*mu: min={min(prod):.2f} max={max(prod):.2f} "
          f"(near-constant => migration speed ~ 1/mu)")

    fig, axes = plt.subplots(1, len(MUS), figsize=(5 * len(MUS), 5))
    for ax, r in zip(axes, results):
        ax.set_facecolor('#0b1021')
        ax.imshow(ramp_field(), cmap='viridis', origin='lower',
                  extent=[0, L, 0, L], alpha=0.4)
        # start (grey) and end (blue) positions
        ax.scatter(r['start'][:, 0], r['start'][:, 1], s=40, c='grey',
                   alpha=0.5, label='start')
        ax.scatter(r['end'][:, 0], r['end'][:, 1], s=40, c='deepskyblue',
                   edgecolors='white', linewidths=0.5, label='end')
        # COM trajectory
        ax.plot(r['com'][:, 0], r['com'][:, 1], '-', color='orange', lw=2)
        ax.annotate('', xy=(r['com'][-1, 0], r['com'][-1, 1]),
                    xytext=(r['start'][:, 0].mean(), r['start'][:, 1].mean()),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"mu={r['mu']:.0f}   COM +x shift = {r['dx']:.2f}", fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle("Directed migration in a chemical gradient vs viscosity "
                 "(gradient increases left->right)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viscosity_migration.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
