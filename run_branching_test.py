import time
from cellflow.cellflow_core import CellSimulation
import numpy as np

np.random.seed(42)

if __name__ == "__main__":
    # Branching morphology test
    #
    # For fingering/branching to occur, the colony must behave like a
    # growing solid with a well-defined interface, NOT a gas of freely
    # migrating cells. This requires:
    #   1. Strong enough adhesion to keep the colony cohesive
    #   2. Low individual motility — growth happens by division at
    #      the colony edge, not by cells running outward
    #   3. Diffusion-limited nutrient supply (Dirichlet BC + slow D)
    #      so only the outermost cells get fed → tips that protrude
    #      into fresh nutrient grow faster → Mullins-Sekerka instability
    #
    # Compare with compact growth: increase nutrient_bc_value to ~50
    # and adhesion_strength to ~10.

    config = {
        'dt': 0.01,
        'physical_size': 500.0,
        'grid_resolution': 500,
        'num_cells': 10,

        # --- Nutrient: Dirichlet BC + very slow diffusion ---
        'nutrient_bc_type': 'dirichlet',
        'nutrient_bc_value': 5.0,        # scarce — strong diffusion limitation
        'nutrient_D': 0.1,               # slow diffusion → sharp depletion zone

        # --- Adhesion: moderate — cohesive colony with a defined edge ---
        'adhesion_strength': 3.0,        # cells stick together (solid-like colony)
        'adhesion_cutoff_factor': 2.5,   # slightly longer range for cohesion
        'repulsion_strength': 50.0,

        # --- Motility: low — colony grows by division, not migration ---
        'walk_speed': 0.02,              # minimal random walk
        'chi_nutrient': 2.0,             # weak chemotaxis — don't scatter outward
        'max_propulsive_force': 10.0,    # gentle push

        # --- Fluid ---
        'viscosity': 50.0,
        'stokeslet_cutoff': 150.0,

        # --- Attractant (off) ---
        'attractant_D': 0.0,
        'chi_attractant': 0.0,

        'enable_biology': True,
        'enable_visualization': True,
    }

    sim = CellSimulation(config, config_name='branching_test')

    start_time = time.time()
    sim.run_simulation(steps=50000, save_interval=500)
    end_time = time.time()

    print(f"\nBranching test completed in {end_time - start_time:.2f} seconds")
    print(f"Final cell count: {len(sim.cells)}")
    if config['enable_visualization']:
        print("Output GIF saved to 'branching_test_simulation.gif'")
    print(f"Data saved to '{sim.output_dir}/' directory.")
