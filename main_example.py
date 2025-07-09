import time
from cellflow_core import CellSimulation
import numpy as np

if __name__ == "__main__":
    # --- Configuration for the main scientific experiment ---
    # This setup is designed to show the full range of behaviors, including
    # growth, division, and hydrodynamic interactions from a central colony.
    config = {
        'dt': 0.05, 
        'domain_size': (300, 300), 
        'num_cells': 20,
        
        # --- Initial and Boundary Conditions ---
        'initial_setup': 'central_uniform', 
        'boundary_condition': 'no_flux',
        'nutrient_bc_value': 20.0,
        'total_nutrient': 5e5, # Used for 'patchy' setup
        'num_patches': 15,     # Used for 'patchy' setup

        # --- Physics and Biology Parameters ---
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

        # --- Flags to control behavior ---
        'enable_biology': True,
        'enable_propulsion': True,
        'constant_force': None,
        'enable_visualization': True, # Set to False for headless runs
    }

    # --- Run Simulation ---
    sim = CellSimulation(config, config_name='main')
    
    start_time = time.time()
    # Run for enough steps for interesting dynamics to emerge
    sim.run_simulation(steps=800, save_interval=100)
    end_time = time.time()

    print(f"Main simulation completed in {end_time - start_time:.2f} seconds")
    if config['enable_visualization']:
        print("Output saved to 'main_simulation.gif'")
    print(f"Data saved to 'simulation_data_main/' directory.")

