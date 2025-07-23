import time
from cellflow.cellflow_core import CellSimulation
import numpy as np

np.random.seed(42) # Use any integer you like, but use the same one for both scripts

if __name__ == "__main__":
    config = {
        'dt': 0.01,
        'physical_size': 3000.0,
        'grid_resolution': 3000,
        'num_cells': 10, # Starting with fewer cells gives them more room to grow and move

        # --- SIMULATION BEHAVIOR FIXES ---
        'nutrient_bc_value': 20.0,      # FIX 1: Much richer nutrient environment
        'walk_speed': 0.05,           # FIX 2: Restore meaningful random motion
        'chi_nutrient': 20.0,         # FIX 3: Restore strong chemotaxis
        'nutrient_D': 0.5,

        # --- Hydrodynamics (kept from previous experiment) ---
        'viscosity': 10.0,            # Lower viscosity to see fluid effects
        'cell_mobility': 1.0,

        # In your config dictionary
        'max_propulsive_force': 50.0, # An example value, in piconewtons for instance

        # --- Other parameters ---
        'adhesion_strength': 0.1,
        'adhesion_cutoff_factor': 2.0,
        'repulsion_strength': 50.0,
        'attractant_D': 0.0,
        'chi_attractant': 0.0,
        'enable_biology': True,
        'enable_visualization': True,
    }

    sim = CellSimulation(config, config_name='Hydro_ON')
    
    start_time = time.time()
    sim.run_simulation(steps=100000, save_interval=100)
    end_time = time.time()

    print(f"\nMain simulation completed in {end_time - start_time:.2f} seconds")
    if config['enable_visualization']:
        print("Output GIF saved to 'main_simulation.gif'")
