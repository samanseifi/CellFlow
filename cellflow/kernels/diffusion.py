"""Scalar-field transport kernels: diffusion and advection."""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def diffuse_field_numba(field, D, dt, dx, bc_value=-1.0):
    """Diffuse a 2D field with configurable boundary conditions.

    bc_value: If < 0 (default), use no-flux (Neumann) boundaries.
              If >= 0, use Dirichlet boundaries held at bc_value
              (models an external reservoir, e.g. petri dish medium).
    """
    new_field = np.copy(field)
    diffusion_factor = D * dt / (dx**2)
    ny, nx = field.shape[0], field.shape[1]
    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            laplacian = (field[i-1, j] + field[i+1, j] +
                         field[i, j-1] + field[i, j+1] - 4 * field[i, j])
            new_field[i, j] += diffusion_factor * laplacian
            if new_field[i, j] < 0:
                new_field[i, j] = 0
    if bc_value >= 0.0:
        # Dirichlet: hold boundary at constant value (external reservoir)
        for i in range(ny):
            new_field[i, 0] = bc_value
            new_field[i, nx - 1] = bc_value
        for j in range(nx):
            new_field[0, j] = bc_value
            new_field[ny - 1, j] = bc_value
    else:
        # Neumann: no-flux (mirror)
        for i in range(ny):
            new_field[i, 0] = new_field[i, 1]
            new_field[i, nx - 1] = new_field[i, nx - 2]
        for j in range(nx):
            new_field[0, j] = new_field[1, j]
            new_field[ny - 1, j] = new_field[ny - 2, j]
    return new_field


@njit(parallel=True, cache=True)
def advect_scalar_field_numba(field, velocity, dt, dx):
    """Advects a scalar field using backward tracing and bilinear interpolation."""
    ny, nx = field.shape
    new_f = np.empty_like(field)

    # Pre-calculate dt/dx to avoid repeated division in the loop
    dt_div_dx = dt / dx

    for j in prange(ny):
        for i in range(nx):
            # Back-trace position in grid-index coordinates
            x = i - velocity[j, i, 0] * dt_div_dx
            y = j - velocity[j, i, 1] * dt_div_dx

            # Clamp to domain boundaries for interpolation
            x = max(0.0, min(nx - 1.000001, x))
            y = max(0.0, min(ny - 1.000001, y))

            # Integer and fractional parts
            i0 = int(x)
            j0 = int(y)
            i1 = i0 + 1
            j1 = j0 + 1

            # Interpolation weights
            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            # Bilinear interpolation
            val = (s0 * (t0 * field[j0, i0] + t1 * field[j1, i0]) +
                   s1 * (t0 * field[j0, i1] + t1 * field[j1, i1]))

            new_f[j, i] = val

    return new_f
