"""End-to-end smoke tests for the CellSimulation engine."""
import os

import numpy as np
import pytest

from cellflow.simulation import CellSimulation


def _base_config(**overrides):
    config = {
        'initial_setup_type': 'central_uniform',
        'num_cells': 8,
        'dt': 0.01,
        'physical_size': 50.0,
        'grid_resolution': 40,
        'nutrient_bc_value': 25.0,
        'nutrient_D': 0.5,
        'chi_nutrient': 15.0,
        'walk_speed': 0.1,
        'max_propulsive_force': 50.0,
        'viscosity': 1e6,
        'cell_mobility': 1.0,
        'adhesion_strength': 0.2,
        'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0,
        'attractant_D': 0.0,
        'chi_attractant': 0.0,
        'enable_visualization': False,
    }
    config.update(overrides)
    return config


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """Run inside a temp dir so output folders/NPZ files don't touch the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_smoke_run_keeps_cells_finite(in_tmp_dir):
    np.random.seed(0)
    sim = CellSimulation(_base_config(), config_name='smoke')
    sim.run_simulation(steps=10, save_interval=5)

    assert len(sim.cells) > 0
    positions = np.array([c.position for c in sim.cells])
    assert np.all(np.isfinite(positions))
    # Fields stay finite (no blow-up).
    assert np.all(np.isfinite(sim.nutrient_field))


def test_cells_stay_within_domain(in_tmp_dir):
    np.random.seed(1)
    sim = CellSimulation(_base_config(), config_name='bounds')
    sim.run_simulation(steps=10, save_interval=100)
    for c in sim.cells:
        assert 0.0 <= c.position[0] <= sim.physical_size
        assert 0.0 <= c.position[1] <= sim.physical_size


def test_npz_snapshot_written(in_tmp_dir):
    np.random.seed(2)
    sim = CellSimulation(_base_config(), config_name='snap')
    sim.run_simulation(steps=6, save_interval=5)
    out = os.path.join('simulation_data_snap', 'snap_data_0005.npz')
    assert os.path.exists(out)
    data = np.load(out, allow_pickle=True)
    assert data['cell_positions'].shape[0] == len(sim.cells)


def test_dirichlet_config_runs(in_tmp_dir):
    """The Dirichlet BC path runs end-to-end. (Edge nodes are not asserted to
    stay pinned: nutrient uptake runs after the diffusion BC reset, so edge
    cells can deplete boundary nodes. The BC itself is covered at the kernel
    level in test_kernels_diffusion.test_dirichlet_holds_boundary_value.)"""
    np.random.seed(3)
    sim = CellSimulation(
        _base_config(nutrient_bc_type='dirichlet', nutrient_bc_value=25.0),
        config_name='dirichlet',
    )
    sim.run_simulation(steps=5, save_interval=100)
    assert np.all(np.isfinite(sim.nutrient_field))


def test_unknown_setup_type_raises(in_tmp_dir):
    with pytest.raises(ValueError):
        CellSimulation(_base_config(initial_setup_type='does_not_exist'), config_name='bad')


def test_brinkman_fft_fluid_model_runs(in_tmp_dir):
    """The Brinkman/IBM fluid path runs end-to-end and stays finite, with cells
    moving with the same field that advects the scalars."""
    np.random.seed(7)
    sim = CellSimulation(
        _base_config(fluid_model='brinkman_fft', viscosity=1.0,
                     brinkman_screening_length=10.0),
        config_name='brinkman',
    )
    assert sim.fluid_model == 'brinkman_fft'
    assert sim.brinkman_alpha > 0.0
    sim.run_simulation(steps=8, save_interval=100)

    positions = np.array([c.position for c in sim.cells])
    assert len(sim.cells) > 0
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(sim.fluid_velocity))
    # Brinkman velocity field is finite everywhere (no ln(r) blow-up).
    assert np.max(np.abs(sim.fluid_velocity)) < 1e6


def test_backward_compat_import():
    """The old monolithic import path must still resolve to the same class."""
    from cellflow.cellflow_core import CellSimulation as Legacy
    from cellflow import CellSimulation as Top
    assert Legacy is CellSimulation is Top
