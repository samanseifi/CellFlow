"""Tests for contact-pressure computation and pressure-inhibited growth
(contact inhibition / homeostatic pressure)."""
import numpy as np

from cellflow.kernels.neighbors import (build_cell_list_numba,
                                        contact_pressure_celllist_numba)
from cellflow.simulation import CellSimulation

L = 100.0
KREP = 30.0


def _pressure(positions, radii):
    pos = np.array(positions, dtype=np.float64)
    rad = np.array(radii, dtype=np.float64)
    bin_size = 2.0 * rad.max()
    order, start, nbx = build_cell_list_numba(pos, L, bin_size)
    return contact_pressure_celllist_numba(pos, rad, KREP, order, start, nbx, bin_size)


# --- the contact-pressure kernel -------------------------------------------

def test_isolated_cells_have_zero_pressure():
    p = _pressure([[20.0, 20.0], [60.0, 60.0]], [3.0, 3.0])
    np.testing.assert_allclose(p, 0.0)


def test_overlapping_pair_has_equal_positive_pressure():
    # centres 5 apart, radii 3+3=6 -> overlap 1
    p = _pressure([[50.0, 50.0], [55.0, 50.0]], [3.0, 3.0])
    assert p[0] > 0.0
    np.testing.assert_allclose(p[0], p[1], rtol=1e-9)


def test_more_overlap_raises_pressure():
    near = _pressure([[50.0, 50.0], [54.0, 50.0]], [3.0, 3.0])[0]   # overlap 2
    far = _pressure([[50.0, 50.0], [55.5, 50.0]], [3.0, 3.0])[0]    # overlap 0.5
    assert near > far


def test_surrounded_cell_feels_more_than_a_single_contact():
    """Isotropic crowding accumulates (net force would cancel, pressure does not)."""
    c = [50.0, 50.0]
    ring = [[50.0 + 5 * np.cos(t), 50.0 + 5 * np.sin(t)]
            for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    pos = [c] + ring
    rad = [3.0] * len(pos)
    p = _pressure(pos, rad)
    pair = _pressure([[50.0, 50.0], [55.0, 50.0]], [3.0, 3.0])[0]
    assert p[0] > pair          # centre cell, many contacts > single contact


# --- integration: pressure-inhibited proliferation -------------------------

def _cfg(**over):
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 30,
        'initial_cluster_radius': 12.0, 'dt': 0.05,
        'physical_size': 90.0, 'grid_resolution': 90,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 95.0,
        'nutrient_D': 1.5, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 4.0,
        'adhesion_strength': 0.2, 'adhesion_cutoff_factor': 1.3,
        'repulsion_strength': 30.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 5,
    }
    cfg.update(over)
    return cfg


def test_pressure_inhibition_limits_proliferation():
    """A well-fed, crowding colony proliferates less when growth is pressure-
    inhibited (its compressed core stops dividing)."""
    free = CellSimulation(_cfg(), config_name='free')
    inhib = CellSimulation(_cfg(enable_pressure_inhibition=True,
                                pressure_threshold=1.5), config_name='inhib')
    for _ in range(120):
        free._simulation_step()
        inhib._simulation_step()
    assert len(inhib.cells) < len(free.cells)
    # the feature actually engaged: some cells exceeded the threshold
    assert max(c.pressure for c in inhib.cells) > 1.5


def test_pressure_defaults_off():
    sim = CellSimulation(_cfg(), config_name='d')
    assert sim.enable_pressure_inhibition is False
    sim._simulation_step()
    # pressure left at the default when the feature is off
    assert all(c.pressure == 0.0 for c in sim.cells)
