"""Effect of viscosity on cell dynamics using the Brinkman/IBM fluid model.

A cluster of cells relaxes under chemotaxis + mechanical forces. With
walk_speed = 0 the stochastic propulsion term vanishes, so the runs are
deterministic and directly comparable across viscosities.

Two things are measured vs viscosity mu (fixed Brinkman screening length):

  1. INITIAL mean cell speed. The configuration is identical at t=0, and for
     Brinkman flow at fixed screening length u = (1/mu) * f/(k^2 + 1/delta^2),
     so the initial speed must scale EXACTLY as 1/mu  ->  speed*mu = const.
     This is a clean validation of the solver's viscosity dependence.

  2. The relaxation trajectory (radius of gyration over time). Higher viscosity
     relaxes proportionally slower.

Produces experiments/viscosity_sweep.png and prints a summary table.

Run:  python experiments/viscosity_sweep.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402


def radius_of_gyration(sim):
    pos = np.array([c.position for c in sim.cells])
    com = pos.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((pos - com) ** 2, axis=1))))


def mean_speed(sim):
    vel = np.array([c.velocity for c in sim.cells])
    return float(np.mean(np.linalg.norm(vel, axis=1)))


def base_config(mu):
    return {
        'initial_setup_type': 'central_uniform',
        'num_cells': 80,
        'dt': 0.01,
        'physical_size': 80.0,
        'grid_resolution': 80,
        'nutrient_bc_value': 20.0,
        'nutrient_D': 0.5,
        'chi_nutrient': 8.0,
        'walk_speed': 0.0,            # deterministic: no random-walk term
        'max_propulsive_force': 20.0,
        'viscosity': mu,
        'adhesion_strength': 0.2,
        'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0,
        'attractant_D': 0.0,
        'chi_attractant': 0.0,
        'enable_visualization': False,
        'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': 20.0,   # fixed -> isolates the 1/mu scaling
    }


def run(mu, steps=120, seed=12345):
    np.random.seed(seed)   # main-thread RNG (cell init, division) is seedable
    sim = CellSimulation(base_config(mu), config_name=f'visc_mu{mu:g}')
    rg = [radius_of_gyration(sim)]
    speed = []
    for _ in range(steps):
        sim._simulation_step()
        speed.append(mean_speed(sim))
        rg.append(radius_of_gyration(sim))
    return {
        'mu': mu,
        'n_cells': len(sim.cells),
        'initial_speed': speed[0],
        'rg': np.array(rg),
        'speed': np.array(speed),
    }


def main():
    mus = [1.0, 4.0, 16.0, 64.0]
    results = [run(mu) for mu in mus]

    print(f"{'mu':>6} {'N':>4} {'initial_speed':>14} {'init_speed*mu':>14}")
    for r in results:
        print(f"{r['mu']:>6.0f} {r['n_cells']:>4d} "
              f"{r['initial_speed']:>14.6f} {r['initial_speed']*r['mu']:>14.4f}")
    prod = [r['initial_speed'] * r['mu'] for r in results]
    spread = (max(prod) - min(prod)) / np.mean(prod)
    print(f"\ninitial_speed*mu spread = {spread*100:.2f}%  "
          f"(~0 confirms initial speed scales exactly as 1/mu, the Stokes law)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        t = np.arange(results[0]['rg'].size)
        for r in results:
            ax[0].plot(t, r['rg'], label=f"mu={r['mu']:.0f}")
            ax[1].semilogy(np.arange(r['speed'].size), r['speed'], label=f"mu={r['mu']:.0f}")
        ax[0].set(xlabel='step', ylabel='radius of gyration',
                  title='Cluster relaxation vs viscosity')
        ax[1].set(xlabel='step', ylabel='mean cell speed (log)',
                  title='Cell speed vs viscosity (~1/mu)')
        for a in ax:
            a.legend()
            a.grid(True, alpha=0.3)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viscosity_sweep.png')
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"\nSaved figure -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    return results


if __name__ == '__main__':
    main()
