import time
from cellflow.cellflow_core import CellSimulation
import numpy as np

np.random.seed(42) # Use any integer you like, but use the same one for both scripts

if __name__ == "__main__":
    config_swimmers = {
        'dt': 0.01,
        'physical_size': 300.0,
        'grid_resolution': 300,
        'num_cells': 50, # A fixed number of swimmers

        # --- CRUCIAL CHANGE ---
        'enable_biology': False, # Turn off growth, division, and consumption

        # --- SETTINGS FOR MOTION ---
        'viscosity': 10.0,       # Low viscosity to allow flow
        'walk_speed': 0.5,       # Give them a reasonable speed
        'chi_nutrient': 0.0,     # Turn off chemotaxis, we only want random swimming

        # --- Other parameters ---
        'cell_mobility': 1.0,
        'adhesion_strength': 0.1,
        'adhesion_cutoff_factor': 2.0,
        'repulsion_strength': 50.0,
        'nutrient_D': 0.0,
        'nutrient_bc_value': 0.0,
        'attractant_D': 0.0,
        'chi_attractant': 0.0,
        'enable_visualization': True,
    }

    sim = CellSimulation(config_swimmers, config_name='Hydro_ON_biology_OFF')

    start_time = time.time()
    sim.run_simulation(steps=5000, save_interval=50)
    end_time = time.time()

    print(f"\nMain simulation completed in {end_time - start_time:.2f} seconds")
    if config_swimmers['enable_visualization']:
        print("Output GIF saved to 'main_simulation.gif'")
