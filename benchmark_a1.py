import time
from cellflow_core import CellSimulation
import numpy as np

if __name__ == "__main__":
    config = {
        'dt': 0.001, 'domain_size': (300, 300), 'num_cells': 1,
        'initial_setup': 'gradient', # Use the specific gradient setup
        'boundary_condition': 'no_flux',
        'chi_nutrient': 50.0, 'chi_attractant': 0.0, 'walk_speed': 0.1,
        'adhesion_strength': 0.0, 'adhesion_cutoff_factor': 1.5, 'viscosity': 0.001,
        'nutrient_D': 0.0, 'attractant_D': 0.0,
        # --- Flags to control behavior ---
        'enable_biology': False, # Turn off division, growth, etc.
        'enable_propulsion': True,
        'constant_force': None,
    }

    sim = CellSimulation(config, config_name='A1')
    start_time = time.time()
    sim.run_simulation(steps=100)
    end_time = time.time()
    print(f"Benchmark A.1 (Chemotaxis) completed in {end_time - start_time:.2f} seconds")