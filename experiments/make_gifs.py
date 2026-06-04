"""Generate animated GIFs of the cluster evolving, for low vs high viscosity.

Uses a compact initial cluster (see viscosity_sweep.base_config) so the
viscosity-dependent spreading is clearly visible. walk_speed=0 keeps the runs
deterministic.

Produces, in the working directory:
  cluster_mu1_simulation.gif   (low viscosity  -> fast spreading)
  cluster_mu64_simulation.gif  (high viscosity -> nearly frozen)

Run:  python experiments/make_gifs.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402
from experiments.viscosity_sweep import base_config  # noqa: E402


def make(mu, steps=120, save_interval=4, seed=12345):
    cfg = base_config(mu)
    cfg['enable_visualization'] = True
    np.random.seed(seed)
    sim = CellSimulation(cfg, config_name=f'cluster_mu{mu:g}')
    sim.run_simulation(steps=steps, save_interval=save_interval)
    print(f"  -> cluster_mu{mu:g}_simulation.gif")


def main():
    for mu in (1.0, 64.0):
        print(f"Rendering mu={mu:g} ...")
        make(mu)


if __name__ == '__main__':
    main()
