"""Tests for the hydrodynamics kernels (cell mobility + fluid velocity)."""
import numpy as np

from cellflow.kernels.stokeslet import (
    compute_cell_velocities_numba,
    update_fluid_velocity_with_dipoles_numba,
)


def test_single_cell_self_mobility_is_stokes_drag():
    """One isolated cell: v = F / (6*pi*mu*R), no hydrodynamic neighbours."""
    positions = np.array([[10.0, 10.0]])
    forces = np.array([[3.0, -2.0]])
    radii = np.array([2.0])
    mu = 5.0
    vel = compute_cell_velocities_numba(positions, forces, radii, viscosity=mu, dx=1.0)

    expected = forces[0] / (6.0 * np.pi * mu * radii[0])
    np.testing.assert_allclose(vel[0], expected, rtol=1e-12)


def test_zero_force_gives_zero_velocity():
    positions = np.array([[5.0, 5.0], [8.0, 5.0]])
    forces = np.zeros((2, 2))
    radii = np.array([2.0, 2.0])
    vel = compute_cell_velocities_numba(positions, forces, radii, viscosity=1.0, dx=1.0)
    np.testing.assert_allclose(vel, 0.0)


def test_pure_dipole_decays_faster_than_monopole_far_field():
    """A force dipole (no net force) produces a faster-decaying flow than a
    single point force, so far from the cell its induced speed is much smaller."""
    grid = 60
    dx = 1.0
    pos = np.array([[30.0, 30.0]])
    orientations = np.array([[1.0, 0.0]])
    dipole_lengths = np.array([4.0])
    propulsive = np.array([[10.0, 0.0]])  # swimming force -> represented as a dipole
    no_mono = np.zeros((1, 2))

    # Dipole-only field
    v_dipole = np.zeros((grid, grid, 2))
    update_fluid_velocity_with_dipoles_numba(
        v_dipole, pos, no_mono, propulsive, orientations, dipole_lengths,
        viscosity=1.0, dx=dx,
    )

    # Monopole field of the same force magnitude (pass it as a monopolar force)
    v_mono = np.zeros((grid, grid, 2))
    update_fluid_velocity_with_dipoles_numba(
        v_mono, pos, propulsive, np.zeros((1, 2)), orientations, dipole_lengths,
        viscosity=1.0, dx=dx,
    )

    # Sample a far-field point.
    gy, gx = 30, 55
    speed_dipole = np.hypot(v_dipole[gy, gx, 0], v_dipole[gy, gx, 1])
    speed_mono = np.hypot(v_mono[gy, gx, 0], v_mono[gy, gx, 1])
    assert speed_dipole < speed_mono
