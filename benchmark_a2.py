import time
from cellflow_core import CellSimulation
import numpy as np

if __name__ == "__main__":
    # --- Configuration for Benchmark A.2: Adhesion ---
    # This benchmark tests if scattered cells correctly aggregate into a single
    # stable colony due to adhesion forces.
    config = {
        'dt': 0.001, 
        'domain_size': (300, 300), 
        'num_cells': 200,
        
        # --- Initial and Boundary Conditions ---
        'initial_setup': 'scattered_uniform',
        'boundary_condition': 'no_flux',
        'nutrient_bc_value': 20.0,
        
        # --- Physics Parameters Tuned for Stable Aggregation ---
        'adhesion_strength': 10.0,          # INCREASED: Makes adhesion much stronger.
        'adhesion_cutoff_factor': 1.8, 
        'repulsion_strength': 1.0,
        'cell_mobility': 5.2,               # INCREASED: Allows forces to have a greater effect.
        'walk_speed': 0.5,                  # DECREASED: Reduces random motion.
        'chi_nutrient': 0.0,                
        'chi_attractant': 0.0, 
        'viscosity': 0.001,                 
        'nutrient_D': 0.0,                  
        'attractant_D': 0.0,

        # --- Flags to control behavior ---
        'enable_biology': False,
        'enable_propulsion': True,
        'constant_force': None,
    }

    # --- Run Simulation ---
    sim = CellSimulation(config, config_name='A2')
    
    start_time = time.time()
    sim.run_simulation(steps=300) # Run for enough steps to see full aggregation
    end_time = time.time()
    
    print(f"Benchmark A.2 (Adhesion) completed in {end_time - start_time:.2f} seconds")
    print("Output saved to 'A2_simulation.gif' and 'simulation_data_A2/' directory.")

