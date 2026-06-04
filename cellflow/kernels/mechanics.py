"""Mechanotransduction kernels: cells sense the local fluid strain-rate.

From the solved velocity field u we compute the velocity-gradient tensor
(du_i/dx_j) on the grid, area-average it over each cell, and derive the
strain-rate magnitude and principal (extensional) axis the cell experiences.
A cell's polarity then aligns nematically toward that axis at a rate set by the
shear magnitude (see simulation step). This closes the fluid->cell loop.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def velocity_gradient_numba(u, dx):
    """Velocity-gradient field by periodic central differences.

    Returns G of shape (ny, nx, 4) with components, in order:
        [ du_x/dx, du_x/dy, du_y/dx, du_y/dy ].
    Periodic in both axes (matches the FFT Brinkman solver's BCs).
    """
    ny, nx, _ = u.shape
    G = np.empty((ny, nx, 4))
    inv2dx = 1.0 / (2.0 * dx)
    for j in prange(ny):
        jm = (j - 1) % ny
        jp = (j + 1) % ny
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            G[j, i, 0] = (u[j, ip, 0] - u[j, im, 0]) * inv2dx   # du_x/dx
            G[j, i, 1] = (u[jp, i, 0] - u[jm, i, 0]) * inv2dx   # du_x/dy
            G[j, i, 2] = (u[j, ip, 1] - u[j, im, 1]) * inv2dx   # du_y/dx
            G[j, i, 3] = (u[jp, i, 1] - u[jm, i, 1]) * inv2dx   # du_y/dy
    return G


@njit(parallel=True, cache=True)
def sample_gradient_at_cells_numba(positions, radii, grad, dx):
    """Area-average the (ny, nx, 4) velocity-gradient field over each cell.

    Returns (n, 4) per-cell averaged gradient components in the same order as
    velocity_gradient_numba. Each cell owns its own output row (race-free).
    """
    n = positions.shape[0]
    ny, nx, _ = grad.shape
    out = np.zeros((n, 4))
    for k in prange(n):
        x_center = int(positions[k, 0] / dx)
        y_center = int(positions[k, 1] / dx)
        r_idx = int(np.ceil(radii[k] / dx))
        r2 = radii[k] * radii[k]
        acc0 = acc1 = acc2 = acc3 = 0.0
        count = 0
        for di in range(-r_idx, r_idx + 1):
            for dj in range(-r_idx, r_idx + 1):
                if (di * dx) ** 2 + (dj * dx) ** 2 <= r2:
                    yy = y_center + dj
                    xx = x_center + di
                    if 0 <= yy < ny and 0 <= xx < nx:
                        acc0 += grad[yy, xx, 0]
                        acc1 += grad[yy, xx, 1]
                        acc2 += grad[yy, xx, 2]
                        acc3 += grad[yy, xx, 3]
                        count += 1
        if count > 0:
            out[k, 0] = acc0 / count
            out[k, 1] = acc1 / count
            out[k, 2] = acc2 / count
            out[k, 3] = acc3 / count
        elif 0 <= y_center < ny and 0 <= x_center < nx:
            out[k, 0] = grad[y_center, x_center, 0]
            out[k, 1] = grad[y_center, x_center, 1]
            out[k, 2] = grad[y_center, x_center, 2]
            out[k, 3] = grad[y_center, x_center, 3]
    return out


def strain_rate_and_axis(grad):
    """Map velocity-gradient components to strain-rate magnitude and principal
    (extensional) axis angle.

    grad: (..., 4) array [du_x/dx, du_x/dy, du_y/dx, du_y/dy].

    Strain-rate tensor E = 0.5 (grad u + grad u^T):
        E_xx = du_x/dx,  E_yy = du_y/dy,  E_xy = 0.5 (du_x/dy + du_y/dx).
    Returns
        shear_rate : sqrt(2 E:E) = sqrt(2(E_xx^2 + E_yy^2 + 2 E_xy^2))
        axis_angle : 0.5 * atan2(2 E_xy, E_xx - E_yy), the extensional direction
                     (a nematic axis, defined mod pi).
    """
    grad = np.asarray(grad, dtype=np.float64)
    Exx = grad[..., 0]
    Eyy = grad[..., 3]
    Exy = 0.5 * (grad[..., 1] + grad[..., 2])
    shear_rate = np.sqrt(2.0 * (Exx**2 + Eyy**2 + 2.0 * Exy**2))
    axis_angle = 0.5 * np.arctan2(2.0 * Exy, Exx - Eyy)
    return shear_rate, axis_angle
