"""Immersed Boundary Method (IBM) coupling between cells and the grid fluid.

Cells exert forces on the fluid grid via a regularized delta kernel
(force *spreading*), and move with the fluid velocity sampled back at their
positions (velocity *interpolation*). Using the SAME regularized delta for both
operations makes the coupling adjoint/consistent, and using the SAME solved
velocity field to move cells and advect scalar fields removes the kinematic
inconsistency of the legacy Stokeslet path.

The kernel is Peskin's 4-point cosine-like delta (support of 4 grid points per
dimension). The grid is treated as periodic, matching the FFT Brinkman solver.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _peskin_phi(r):
    """Peskin 4-point delta weight for a 1D offset r (in grid units)."""
    ar = abs(r)
    if ar >= 2.0:
        return 0.0
    if ar >= 1.0:
        return (5.0 - 2.0 * ar - np.sqrt(-7.0 + 12.0 * ar - 4.0 * ar * ar)) / 8.0
    return (3.0 - 2.0 * ar + np.sqrt(1.0 + 4.0 * ar - 4.0 * ar * ar)) / 8.0


@njit(cache=True)
def spread_forces_numba(positions, forces, ny, nx, dx):
    """Spread point forces onto a periodic grid force-density field.

    force_density(x) = sum_k F_k * delta_h(x - X_k),  delta_h = (1/dx^2) phi*phi

    so that the discrete integral sum(force_density)*dx^2 == sum(forces).
    """
    fd = np.zeros((ny, nx, 2))
    inv_area = 1.0 / (dx * dx)
    for k in range(positions.shape[0]):
        gx = positions[k, 0] / dx
        gy = positions[k, 1] / dx
        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        for di in range(-1, 3):
            ix = i0 + di
            wx = _peskin_phi(gx - ix)
            if wx == 0.0:
                continue
            ixw = ix % nx
            for dj in range(-1, 3):
                iy = j0 + dj
                wy = _peskin_phi(gy - iy)
                if wy == 0.0:
                    continue
                iyw = iy % ny
                w = wx * wy * inv_area
                fd[iyw, ixw, 0] += forces[k, 0] * w
                fd[iyw, ixw, 1] += forces[k, 1] * w
    return fd


@njit(parallel=True, cache=True)
def interpolate_velocity_numba(u, positions, dx):
    """Interpolate a periodic grid velocity field back to cell positions.

    U_k = sum_x u(x) * phi*phi   (weights sum to 1; a partition-of-unity average)
    """
    n = positions.shape[0]
    ny, nx, _ = u.shape
    vel = np.zeros((n, 2))
    for k in prange(n):
        gx = positions[k, 0] / dx
        gy = positions[k, 1] / dx
        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        vx = 0.0
        vy = 0.0
        for di in range(-1, 3):
            ix = i0 + di
            wx = _peskin_phi(gx - ix)
            if wx == 0.0:
                continue
            ixw = ix % nx
            for dj in range(-1, 3):
                iy = j0 + dj
                wy = _peskin_phi(gy - iy)
                if wy == 0.0:
                    continue
                iyw = iy % ny
                w = wx * wy
                vx += u[iyw, ixw, 0] * w
                vy += u[iyw, ixw, 1] * w
        vel[k, 0] = vx
        vel[k, 1] = vy
    return vel
