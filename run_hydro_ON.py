import time
from cellflow_core import CellSimulation
import numpy as np

if __name__ == "__main__":
    # --- Configuration for the Hydrodynamics ON Test ---
    # This simulation includes the effect of long-range fluid flows.
    config = {
        'dt': 0.05, 
        'domain_size': (300, 300), 
        'num_cells': 200,
        
        # --- Initial and Boundary Conditions ---
        'initial_setup': 'central_uniform',
        'boundary_condition': 'no_flux',
        'nutrient_bc_value': 20.0,
        
        # --- Physics Parameters ---
        'adhesion_strength': 1.0, 
        'adhesion_cutoff_factor': 1.8, 
        'repulsion_strength': 50.0,
        'cell_mobility': 0.1,
        'walk_speed': 0.2,
        'chi_nutrient': 10.0, 
        'chi_attractant': 5.0, 
        'viscosity': 10.0, # MODERATE viscosity allows for fluid flow.
        'nutrient_D': 1.0, 
        'attractant_D': 2.0,

        # --- Flags to control behavior ---
        'enable_biology': True,
        'enable_propulsion': True,
        'constant_force': None,
        'enable_visualization': False, # Set to False for headless runs
    }

    # --- Run Simulation ---
    sim = CellSimulation(config, config_name='Hydro_ON')
    
    start_time = time.time()
    sim.run_simulation(steps=1200)
    end_time = time.time()
    
    print(f"Hydrodynamics ON simulation completed in {end_time - start_time:.2f} seconds")
    print("Output saved to 'Hydro_ON_simulation.gif' and 'simulation_data_Hydro_ON/' directory.")

