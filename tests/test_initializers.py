"""Tests for the initial-condition factories."""
import numpy as np

from cellflow.initializers import (
    setup_central_uniform,
    setup_wound_healing,
    INITIALIZER_MAP,
)


def test_initializer_map_keys():
    assert set(INITIALIZER_MAP) == {'central_uniform', 'wound_healing'}


def test_central_uniform_cell_count_and_field_shape():
    config = {'num_cells': 12, 'nutrient_bc_value': 20.0}
    cells, field = setup_central_uniform(config, physical_size=100.0, grid_resolution=50)
    assert len(cells) == 12
    assert field.shape == (50, 50)
    np.testing.assert_allclose(field, 20.0)


def test_wound_healing_leaves_a_gap():
    """No cell centers should fall inside the wound gap band."""
    physical_size = 200.0
    gap = 80.0
    config = {'wound_gap_width': gap, 'initial_cell_spacing': 5.0, 'nutrient_bc_value': 100.0}
    cells, field = setup_wound_healing(config, physical_size, grid_resolution=100)

    left = physical_size / 2 - gap / 2
    right = physical_size / 2 + gap / 2
    xs = np.array([c.position[0] for c in cells])

    assert len(cells) > 0
    assert np.all((xs < left) | (xs >= right))
    # Cells populate both sides of the wound.
    assert np.any(xs < left)
    assert np.any(xs >= right)
