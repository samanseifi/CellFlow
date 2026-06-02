"""Persistence of simulation state to compressed NPZ snapshots."""
import os

import numpy as np


def save_data_npz(cells, nutrient_field, attractant_field, fluid_velocity,
                  config, output_dir, config_name, step):
    """Save the simulation state at a specific step to a compressed NPZ file."""
    if not cells:
        return  # Don't save if there are no cells

    cell_positions = np.array([cell.position for cell in cells])
    cell_velocities = np.array([cell.velocity for cell in cells])
    cell_radii = np.array([cell.radius for cell in cells])
    cell_ids = np.array([cell.id for cell in cells])
    cell_nutrients = np.array([cell.nutrient_accumulated for cell in cells])

    filepath = os.path.join(output_dir, f'{config_name}_data_{step:04d}.npz')

    np.savez_compressed(
        filepath,
        step=step,
        cell_positions=cell_positions,
        cell_velocities=cell_velocities,
        cell_radii=cell_radii,
        cell_ids=cell_ids,
        cell_nutrients=cell_nutrients,
        final_nutrient_field=nutrient_field,
        final_attractant_field=attractant_field,
        final_fluid_velocity=fluid_velocity,
        config=config,
    )
