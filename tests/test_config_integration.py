"""Integration tests for the new config-level wiring: diffusion-solver selection,
saturating-uptake propagation to cells, and trait inheritance on division."""
import numpy as np
import pytest

from cellflow.cell import Cell
from cellflow.simulation import CellSimulation
from cellflow.kernels.diffusion import (diffuse_field_numba,
                                        diffuse_field_implicit_numba)


def _cfg(**over):
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 12,
        'initial_cluster_radius': 6.0, 'dt': 0.05,
        'physical_size': 60.0, 'grid_resolution': 60,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 80.0,
        'nutrient_D': 1.0, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 4.0,
        'adhesion_strength': 0.2, 'adhesion_cutoff_factor': 1.3,
        'repulsion_strength': 30.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 3,
    }
    cfg.update(over)
    return cfg


# --- diffusion solver selection --------------------------------------------

def test_default_solver_is_explicit():
    sim = CellSimulation(_cfg(), config_name='t')
    assert sim.diffusion_solver == 'explicit'
    assert sim._diffuse is diffuse_field_numba


def test_implicit_solver_selected_by_config():
    sim = CellSimulation(_cfg(diffusion_solver='implicit'), config_name='t')
    assert sim._diffuse is diffuse_field_implicit_numba


def test_invalid_solver_raises():
    with pytest.raises(ValueError):
        CellSimulation(_cfg(diffusion_solver='banana'), config_name='t')


def test_implicit_solver_runs_through_simulation():
    sim = CellSimulation(_cfg(diffusion_solver='implicit', nutrient_D=4.0),
                         config_name='t')
    for _ in range(15):
        sim._simulation_step()
    assert np.all(np.isfinite(sim.nutrient_field))
    assert sim.nutrient_field.min() >= 0.0


# --- saturating uptake propagation -----------------------------------------

def test_saturation_config_applied_to_all_cells():
    sim = CellSimulation(_cfg(nutrient_uptake_saturation=20.0), config_name='t')
    assert all(c.uptake_saturation == 20.0 for c in sim.cells)


def test_default_uptake_is_linear_on_cells():
    sim = CellSimulation(_cfg(), config_name='t')
    assert all(c.uptake_saturation <= 0.0 for c in sim.cells)


def test_saturating_uptake_depletes_less_through_sim():
    """Through the batched biology kernel, Monod uptake (Km < field) removes less
    nutrient than the linear law, so the colony interior stays better fed."""
    lin = CellSimulation(_cfg(), config_name='lin')
    sat = CellSimulation(_cfg(nutrient_uptake_saturation=20.0), config_name='sat')
    for _ in range(30):
        lin._simulation_step()
        sat._simulation_step()
    c = 30  # centre index (grid 60)
    assert sat.nutrient_field[c, c] > lin.nutrient_field[c, c]


# --- trait inheritance on division -----------------------------------------

def test_daughter_inherits_uptake_saturation():
    Cell.next_id = 0
    c = Cell([10.0, 10.0], nutrient=100.0, area_conserving=True)
    c.uptake_saturation = 12.5
    c.update_phase()                 # -> DIVISION (radius at max for nutrient=100)
    d = c.divide()
    assert d is not None
    assert d.uptake_saturation == 12.5
