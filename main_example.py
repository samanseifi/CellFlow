import time
from cellflow.cellflow_core import CellSimulation
import numpy as np

if __name__ == "__main__":
    # --- Configuration for the main scientific experiment ---
    config = {
        'dt': 0.05, 
        'domain_size': (300, 300), # Using a larger domain for better visuals
        'num_cells': 20,
        
        'initial_setup': 'central_uniform', 
        'nutrient_bc_value': 20.0,

        'adhesion_strength': 1.0, 
        'adhesion_cutoff_factor': 1.8, 
        'repulsion_strength': 50.0,
        'cell_mobility': 0.1,
        'walk_speed': 0.2,
        'chi_nutrient': 10.0, 
        'chi_attractant': 5.0, 
        'viscosity': 10.0,
        'nutrient_D': 1.0, 
        'attractant_D': 2.0,

        'enable_biology': True,
        'enable_visualization': True,
    }

    # --- Run Simulation ---
    sim = CellSimulation(config, config_name='main')
    
    start_time = time.time()
    # Run for enough steps for interesting dynamics to emerge
    sim.run_simulation(steps=1600, save_interval=50) # Reduced interval for smoother GIF
    end_time = time.time()

    print(f"\nMain simulation completed in {end_time - start_time:.2f} seconds")
    if config['enable_visualization']:
        print("Output saved to 'main_simulation.gif'")
    print(f"Data saved to '{sim.output_dir}/' directory.")

