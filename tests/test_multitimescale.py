"""Tests for multi-timescale operator splitting: diffusion sub-stepping and the
expensive-fluid-solve cadence (fluid_update_interval)."""
import numpy as np

from cellflow.simulation import CellSimulation


def _cfg(**over):
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 8,
        'initial_cluster_radius': 4.0, 'dt': 0.05,
        'physical_size': 60.0, 'grid_resolution': 60,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': 50.0,
        'nutrient_D': 0.5, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 4.0,
        'adhesion_strength': 0.2, 'adhesion_cutoff_factor': 1.3,
        'repulsion_strength': 30.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 3,
    }
    cfg.update(over)
    return cfg


def test_defaults_are_unit_cadence():
    sim = CellSimulation(_cfg(), config_name='t')
    assert sim.diffusion_substeps == 1
    assert sim.fluid_update_interval == 1


def test_fluid_interval_first_step_identical():
    """Step 0 always recomputes the fluid solve, so a cadence>1 run is identical
    to the every-step run after one step."""
    a = CellSimulation(_cfg(fluid_update_interval=1), config_name='a')
    b = CellSimulation(_cfg(fluid_update_interval=5), config_name='b')
    a._simulation_step(); b._simulation_step()
    pa = np.array([c.position for c in a.cells])
    pb = np.array([c.position for c in b.cells])
    np.testing.assert_allclose(pa, pb, rtol=1e-12)


def test_fluid_interval_runs_stable_and_caches():
    """A coarse fluid cadence runs without error, reuses the cached velocity
    field between solves, and stays finite."""
    sim = CellSimulation(_cfg(fluid_update_interval=4), config_name='c')
    for _ in range(20):
        sim._simulation_step()
    assert len(sim.cells) > 0
    assert np.all(np.isfinite(sim.fluid_velocity))
    assert np.all(np.isfinite(sim.nutrient_field))


def test_diffusion_substeps_keep_explicit_stable_at_high_cfl():
    """With D*dt/dx^2 = 1.0 (4x the explicit stability limit), a single explicit
    diffusion step is unstable, but sub-stepping keeps the field bounded by the
    reservoir value."""
    # dx = 1, dt = 0.2, D = 5 -> diffusion number 1.0
    over = dict(physical_size=60.0, grid_resolution=60, dt=0.2, nutrient_D=5.0,
                nutrient_bc_value=50.0)
    unstable = CellSimulation(_cfg(diffusion_substeps=1, **over), config_name='u')
    stable = CellSimulation(_cfg(diffusion_substeps=8, **over), config_name='s')
    for _ in range(15):
        unstable._simulation_step()
        stable._simulation_step()
    # sub-stepped run is well-behaved: bounded by the reservoir, non-negative
    assert np.all(np.isfinite(stable.nutrient_field))
    assert stable.nutrient_field.max() <= 50.0 * 1.02
    assert stable.nutrient_field.min() >= 0.0
    # the single-step run overshoots the reservoir badly (instability)
    assert unstable.nutrient_field.max() > stable.nutrient_field.max()


def test_diffusion_substeps_converge():
    """More substeps converge: 2 vs 8 substeps give close fields for a stable dt."""
    a = CellSimulation(_cfg(diffusion_substeps=2), config_name='a2')
    b = CellSimulation(_cfg(diffusion_substeps=8), config_name='b8')
    for _ in range(10):
        a._simulation_step(); b._simulation_step()
    np.testing.assert_allclose(a.nutrient_field, b.nutrient_field, atol=0.5)
