import cProfile
import pstats
import time
from cellflow_core import CellSimulation
import numpy as np

def profile_simulation():
    """
    Runs a short simulation under the cProfile profiler to find bottlenecks.
    """
    # Use a configuration that is known to be demanding
    config = {
        'dt': 0.05, 
        'domain_size': (200, 200), # Smaller domain for a quicker profile
        'num_cells': 100,
        'initial_setup': 'central_uniform',
        'boundary_condition': 'no_flux',
        'nutrient_bc_value': 20.0,
        'adhesion_strength': 2.0, 
        'adhesion_cutoff_factor': 1.8, 
        'repulsion_strength': 50.0,
        'cell_mobility': 0.1,
        'walk_speed': 0.5,
        'chi_nutrient': 10.0, 
        'chi_attractant': 5.0, 
        'viscosity': 10.0,
        'nutrient_D': 1.0, 
        'attractant_D': 2.0,
        'enable_biology': True,
        'enable_propulsion': True,
        'constant_force': None,
    }

    # Create the simulation object
    sim = CellSimulation(config, config_name='profile_run')

    # Define the function to profile
    def run_short():
        # We only need a few steps to see where the time is spent
        sim.run_simulation(steps=10)

    # --- Run the Profiler ---
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_short()
    
    profiler.disable()

    # --- Print the Stats ---
    print("--- Simulation Performance Profile ---")
    # Sort the stats by cumulative time spent in each function
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20) # Print the top 20 most time-consuming functions

if __name__ == "__main__":
    profile_simulation()

