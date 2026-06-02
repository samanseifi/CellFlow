"""Tests for differential (type-dependent) adhesion."""
import numpy as np
import pytest

from cellflow.kernels.forces import calculate_adhesion_forces_numba
from cellflow.kernels.adhesion import calculate_differential_adhesion_forces_numba
from cellflow.simulation import CellSimulation


def test_uniform_matrix_matches_scalar_adhesion():
    """A constant adhesion matrix must reproduce the legacy scalar kernel exactly."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 30, size=(20, 2))
    radii = np.full(20, 2.0)
    types = rng.integers(0, 3, size=20).astype(np.int64)
    strength, cutoff = 0.7, 1.5

    matrix = np.full((3, 3), strength)
    diff = calculate_differential_adhesion_forces_numba(
        positions, radii, types, matrix, cutoff
    )
    scalar = calculate_adhesion_forces_numba(positions, radii, strength, cutoff)
    np.testing.assert_allclose(diff, scalar, atol=1e-12)


def test_stronger_self_adhesion_pulls_harder():
    """Same-type neighbours with a deeper well exert a larger attractive force."""
    # i at origin, a same-type neighbour within the adhesion band.
    positions = np.array([[0.0, 0.0], [5.0, 0.0]])
    radii = np.array([2.0, 2.0])           # touch 4, cutoff 6 -> 5 is in band
    types = np.array([0, 0], dtype=np.int64)

    weak = np.array([[1.0, 0.0], [0.0, 1.0]])
    strong = np.array([[3.0, 0.0], [0.0, 1.0]])
    f_weak = calculate_differential_adhesion_forces_numba(positions, radii, types, weak, 1.5)
    f_strong = calculate_differential_adhesion_forces_numba(positions, radii, types, strong, 1.5)

    # Both attract toward +x; the stronger self-adhesion pulls harder.
    assert f_strong[0, 0] > f_weak[0, 0] > 0.0


def test_cross_type_can_be_nonadhesive():
    """Zero cross-adhesion -> no force between different types."""
    positions = np.array([[0.0, 0.0], [5.0, 0.0]])
    radii = np.array([2.0, 2.0])
    types = np.array([0, 1], dtype=np.int64)
    matrix = np.array([[2.0, 0.0], [0.0, 2.0]])  # cohesive, non-adhesive across
    f = calculate_differential_adhesion_forces_numba(positions, radii, types, matrix, 1.5)
    np.testing.assert_allclose(f, 0.0)


def test_simulation_left_right_type_assignment():
    config = {
        'initial_setup_type': 'central_uniform', 'num_cells': 30, 'dt': 0.01,
        'physical_size': 50.0, 'grid_resolution': 32, 'nutrient_bc_value': 25.0,
        'nutrient_D': 0.5, 'chi_nutrient': 0.0, 'walk_speed': 0.0,
        'max_propulsive_force': 0.0, 'viscosity': 1e6, 'cell_mobility': 1.0,
        'adhesion_cutoff_factor': 1.5, 'repulsion_strength': 50.0,
        'attractant_D': 0.0, 'chi_attractant': 0.0, 'enable_visualization': False,
        'adhesion_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'cell_type_assignment': 'left_right',
    }
    np.random.seed(0)
    sim = CellSimulation(config, config_name='types')
    mid = sim.physical_size / 2
    for cell in sim.cells:
        expected = 0 if cell.position[0] < mid else 1
        assert cell.cell_type == expected


def test_invalid_adhesion_matrix_raises():
    config = {
        'initial_setup_type': 'central_uniform', 'num_cells': 4, 'dt': 0.01,
        'physical_size': 50.0, 'grid_resolution': 16, 'nutrient_bc_value': 25.0,
        'nutrient_D': 0.5, 'chi_nutrient': 0.0, 'walk_speed': 0.0,
        'max_propulsive_force': 0.0, 'viscosity': 1e6, 'cell_mobility': 1.0,
        'adhesion_cutoff_factor': 1.5, 'repulsion_strength': 50.0,
        'attractant_D': 0.0, 'chi_attractant': 0.0, 'enable_visualization': False,
        'adhesion_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # not square
    }
    with pytest.raises(ValueError):
        CellSimulation(config, config_name='badmatrix')


def _same_type_neighbor_fraction(sim, neighbor_radius):
    """Average fraction of in-range neighbours that share a cell's type."""
    pos = np.array([c.position for c in sim.cells])
    typ = np.array([c.cell_type for c in sim.cells])
    fracs = []
    for i in range(len(pos)):
        d = np.linalg.norm(pos - pos[i], axis=1)
        nb = (d > 0) & (d < neighbor_radius)
        if nb.sum() > 0:
            fracs.append(np.mean(typ[nb] == typ[i]))
    return float(np.mean(fracs)) if fracs else 0.0


@pytest.mark.slow
def test_two_types_sort_over_time(tmp_path, monkeypatch):
    """With strong cohesion and zero cross-adhesion, like cells should cluster:
    the same-type neighbour fraction increases relative to the initial mix."""
    monkeypatch.chdir(tmp_path)
    config = {
        'initial_setup_type': 'central_uniform', 'num_cells': 120, 'dt': 0.02,
        'physical_size': 60.0, 'grid_resolution': 48, 'nutrient_bc_value': 25.0,
        'nutrient_D': 0.5, 'chi_nutrient': 0.0, 'walk_speed': 0.5,
        'max_propulsive_force': 5.0, 'viscosity': 1.0, 'cell_mobility': 1.0,
        'adhesion_cutoff_factor': 2.0, 'repulsion_strength': 50.0,
        'attractant_D': 0.0, 'chi_attractant': 0.0, 'enable_visualization': False,
        'adhesion_matrix': [[4.0, 0.0], [0.0, 4.0]],
        'cell_type_assignment': 'random', 'cell_type_fractions': [0.5, 0.5],
    }
    np.random.seed(1)
    sim = CellSimulation(config, config_name='sort')
    before = _same_type_neighbor_fraction(sim, neighbor_radius=10.0)
    sim.run_simulation(steps=120, save_interval=1000)
    after = _same_type_neighbor_fraction(sim, neighbor_radius=10.0)
    assert after > before
