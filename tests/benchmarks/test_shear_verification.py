"""Physics verification of mechanotransduction (issue #17).

Under a known steady simple-shear flow u = (gamma * y, 0):
  - the strain rate cells sense equals gamma,
  - the principal (extensional) strain axis is +45 degrees,
  - a cell's polarity, evolving by nematic alignment, converges to that axis.

This is verified end-to-end through CellSimulation._update_polarity (the same
code path the solver uses), with the velocity field imposed analytically so the
expected answer is exact.
"""
import numpy as np
import pytest

from cellflow.simulation import CellSimulation
from cellflow.kernels.mechanics import (
    sample_gradient_at_cells_numba, velocity_gradient_numba, strain_rate_and_axis,
)

G = 80
L = 40.0
DX = L / G
GAMMA = 1.0


def _sim(tmp_path, monkeypatch, **over):
    monkeypatch.chdir(tmp_path)
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 6,
        'initial_cluster_radius': 1.5, 'dt': 0.02,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_value': 20.0, 'nutrient_D': 0.5, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 0.0, 'viscosity': 1.0,
        'adhesion_strength': 0.0, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 0.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'seed': 0,
        'enable_mechanotransduction': True, 'shear_alignment_rate': 2.0,
    }
    cfg.update(over)
    return CellSimulation(cfg, config_name='shear')


def _simple_shear_field():
    """u_x = gamma * y (row j -> y = j*dx); u_y = 0. Linear -> central diff exact."""
    y = (np.arange(G) * DX)[:, None]
    u = np.zeros((G, G, 2))
    u[:, :, 0] = GAMMA * y
    return u


def test_sensed_shear_rate_and_axis_are_correct(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
    # place cells at the center (away from the periodic y-wrap rows)
    for c in sim.cells:
        c.position = np.array([L / 2, L / 2])
    sim.fluid_velocity = _simple_shear_field()
    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])
    grad = velocity_gradient_numba(sim.fluid_velocity, DX)
    cell_grad = sample_gradient_at_cells_numba(pos, rad, grad, DX)
    shear, axis = strain_rate_and_axis(cell_grad)
    np.testing.assert_allclose(shear, GAMMA, atol=1e-6)
    np.testing.assert_allclose(axis, np.pi / 4, atol=1e-6)


def test_polarity_converges_to_strain_axis(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
    for c in sim.cells:
        c.position = np.array([L / 2, L / 2])
        c.polarity = 0.0                      # start misaligned
    sim.fluid_velocity = _simple_shear_field()
    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])

    for _ in range(2000):
        sim._update_polarity(pos, rad)

    pol = np.array([c.polarity for c in sim.cells])
    np.testing.assert_allclose(pol, np.pi / 4, atol=1e-2)


def test_zero_shear_leaves_polarity_unchanged(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
    for c in sim.cells:
        c.position = np.array([L / 2, L / 2])
        c.polarity = 0.3
    sim.fluid_velocity = np.zeros((G, G, 2))   # no flow -> no shear -> no torque
    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])
    for _ in range(50):
        sim._update_polarity(pos, rad)
    pol = np.array([c.polarity for c in sim.cells])
    np.testing.assert_allclose(pol, 0.3, atol=1e-12)
