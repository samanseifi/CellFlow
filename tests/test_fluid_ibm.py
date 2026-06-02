"""Tests for the Immersed Boundary spreading / interpolation kernels."""
import numpy as np

from cellflow.fluid.ibm import spread_forces_numba, interpolate_velocity_numba


def test_spread_conserves_total_force():
    """The discrete integral of the spread force density equals the total of the
    point forces: sum(force_density) * dx^2 == sum(forces)."""
    dx = 0.5
    ny = nx = 40
    positions = np.array([[5.0, 5.0], [12.3, 8.7], [3.0, 15.1]])
    forces = np.array([[1.0, -2.0], [0.5, 0.5], [-1.5, 1.0]])
    fd = spread_forces_numba(positions, forces, ny, nx, dx)

    total = fd.sum(axis=(0, 1)) * dx * dx
    np.testing.assert_allclose(total, forces.sum(axis=0), atol=1e-10)


def test_interpolation_of_uniform_field_is_exact():
    """Interpolation weights form a partition of unity, so a uniform field is
    recovered exactly at any cell position."""
    dx = 0.5
    ny = nx = 32
    u = np.empty((ny, nx, 2))
    u[:, :, 0] = 3.0
    u[:, :, 1] = -1.0
    positions = np.array([[4.2, 7.9], [10.0, 10.0], [1.1, 14.4]])
    vel = interpolate_velocity_numba(u, positions, dx)
    np.testing.assert_allclose(vel[:, 0], 3.0, atol=1e-12)
    np.testing.assert_allclose(vel[:, 1], -1.0, atol=1e-12)


def test_spread_then_interpolate_is_symmetric_pairing():
    """Spreading is the adjoint of interpolation: for any grid field g and point
    force F at X,  <spread(F), g>_grid * dx^2 == F . interp(g)(X).
    This is the discrete adjoint identity that makes the IBM coupling consistent."""
    dx = 1.0
    ny = nx = 24
    rng = np.random.default_rng(0)
    g = rng.standard_normal((ny, nx, 2))
    X = np.array([[10.4, 12.7]])
    F = np.array([[2.0, -1.0]])

    fd = spread_forces_numba(X, F, ny, nx, dx)
    lhs = np.sum(fd * g) * dx * dx                 # <spread(F), g> * dx^2
    interp = interpolate_velocity_numba(g, X, dx)  # interp(g)(X)
    rhs = F[0] @ interp[0]
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)
