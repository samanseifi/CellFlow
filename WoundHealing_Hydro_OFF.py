# Save this as run_simulation.py
import time
import numpy as np
from cellflow.cellflow_core import CellSimulation # Assuming your core file is in cellflow/cellflow_core.py

# Set a seed for reproducibility
np.random.seed(42)

if __name__ == "__main__":
    # --- COMPLETE INPUT PARAMETERS ---
    config = {
        # --- Simulation Control ---
        'initial_setup_type': 'wound_healing', # Use the wound healing initializer
        'dt': 0.01,
        'enable_biology': True,
        'enable_visualization': True,

        # --- Domain and Grid ---
        'physical_size': 300.0,
        'grid_resolution': 300,

        # --- Biological Parameters ---
        'nutrient_bc_value': 25.0,           # FIX: Reduced to prevent explosive growth
        'nutrient_D': 0.5,
        'chi_nutrient': 15.0,

        # --- Cell Motility (Constant Force Model) ---
        'walk_speed': 0.1,
        'max_propulsive_force': 50.0,

        # --- Physics and Hydrodynamics ---
        'viscosity': 1000000.0,                   # "Hydro ON" to see instabilities
        'cell_mobility': 1.0,

        # --- Inter-Cell Mechanical Forces ---
        'adhesion_strength': 0.2,
        'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0,

        # --- Wound Healing Specific Parameters ---
        'wound_gap_width': 80.0,
        'initial_cell_spacing': 5.0,

        # --- Unused Parameters ---
        'attractant_D': 0.0,
        'chi_attractant': 0.0,
    }

    # --- Run Simulation ---
    # The config_name helps create separate output folders
    sim = CellSimulation(config, config_name='WoundHealing_Hydro_OFF')
    
    start_time = time.time()
    sim.run_simulation(steps=1100, save_interval=100)
    end_time = time.time()

    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")