"""Initial-condition factories that build the cell population and nutrient field.

Register new setups in INITIALIZER_MAP; CellSimulation selects one via the
``initial_setup_type`` config key.
"""
import numpy as np

from .cell import Cell


def setup_central_uniform(config, physical_size, grid_resolution):
    """Initializes cells in a cluster at the center."""
    nutrient_field = np.ones((grid_resolution, grid_resolution)) * config.get('nutrient_bc_value', 20.0)
    center = physical_size / 2
    initial_cluster_radius = config.get('initial_cluster_radius', 20.0)
    Cell.next_id = 0
    cells = [Cell(np.array([center, center]) + np.random.randn(2) * initial_cluster_radius) for _ in range(config['num_cells'])]
    return cells, nutrient_field


def setup_wound_healing(config, physical_size, grid_resolution):
    """Creates an initial state simulating a wound healing assay."""
    nutrient_field = np.ones((grid_resolution, grid_resolution)) * config.get('nutrient_bc_value', 100.0)
    cells = []
    Cell.next_id = 0

    wound_gap = config.get('wound_gap_width', 80.0)
    left_boundary = (physical_size / 2) - (wound_gap / 2)
    right_boundary = (physical_size / 2) + (wound_gap / 2)

    cell_spacing = config.get('initial_cell_spacing', 5.0)
    y_positions = np.arange(cell_spacing, physical_size, cell_spacing)

    # Left sheet
    x_positions_left = np.arange(cell_spacing, left_boundary, cell_spacing)
    for x in x_positions_left:
        for y in y_positions:
            cells.append(Cell(np.array([x, y])))

    # Right sheet
    x_positions_right = np.arange(right_boundary, physical_size, cell_spacing)
    for x in x_positions_right:
        for y in y_positions:
            cells.append(Cell(np.array([x, y])))

    return cells, nutrient_field


# Maps a string name to the setup function.
INITIALIZER_MAP = {
    'central_uniform': setup_central_uniform,
    'wound_healing': setup_wound_healing,
}
