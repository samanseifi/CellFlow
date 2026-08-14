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


@njit(cache=True)
def _thomas(a, b, c, d):
    """Solve a tridiagonal system (Thomas algorithm), O(n). a=sub, b=diag,
    c=super, d=rhs (a[0], c[n-1] unused). Returns the solution vector."""
    n = b.shape[0]
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, n):
        m = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / m
        dp[k] = (d[k] - a[k] * dp[k - 1]) / m
    x = np.empty(n)
    x[n - 1] = dp[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


@njit(parallel=True, cache=True)
def diffuse_field_implicit_numba(field, D, dt, dx, bc_value=-1.0):
    """Diffuse a 2D field with the **implicit ADI** (Peaceman-Rachford) scheme.

    Unconditionally stable (any dt), second-order in space and time, and -- unlike
    the explicit solver -- it preserves the correct steady state exactly, so it
    reaches the quasi-steady nutrient profile in a few large steps regardless of
    D. Each half-step is a set of independent tridiagonal solves (Thomas, O(N)),
    parallelized over lines.

    bc_value: < 0 -> no-flux (Neumann) boundaries; >= 0 -> Dirichlet at bc_value.
    Same convention/signature as ``diffuse_field_numba`` so the two are drop-in
    interchangeable.
    """
    ny, nx = field.shape[0], field.shape[1]
    r = D * dt / (dx * dx)
    h = 0.5 * r
    dirichlet = bc_value >= 0.0
    ustar = np.copy(field)
    new_field = np.copy(field)

    if dirichlet:
        # ---- half 1: implicit in x, explicit in y; interior rows 1..ny-2 ----
        m = nx - 2
        for i in prange(1, ny - 1):
            a = np.full(m, -h)
            b = np.full(m, 1.0 + r)
            c = np.full(m, -h)
            d = np.empty(m)
            for k in range(m):
                j = k + 1
                d[k] = field[i, j] + h * (field[i - 1, j] - 2.0 * field[i, j] + field[i + 1, j])
            d[0] += h * bc_value
            d[m - 1] += h * bc_value
            sol = _thomas(a, b, c, d)
            for k in range(m):
                ustar[i, k + 1] = sol[k]
            ustar[i, 0] = bc_value
            ustar[i, nx - 1] = bc_value
        for j in range(nx):
            ustar[0, j] = bc_value
            ustar[ny - 1, j] = bc_value
        # ---- half 2: implicit in y, explicit in x; interior cols 1..nx-2 ----
        mm = ny - 2
        for j in prange(1, nx - 1):
            a = np.full(mm, -h)
            b = np.full(mm, 1.0 + r)
            c = np.full(mm, -h)
            d = np.empty(mm)
            for k in range(mm):
                i = k + 1
                d[k] = ustar[i, j] + h * (ustar[i, j - 1] - 2.0 * ustar[i, j] + ustar[i, j + 1])
            d[0] += h * bc_value
            d[mm - 1] += h * bc_value
            sol = _thomas(a, b, c, d)
            for k in range(mm):
                new_field[k + 1, j] = sol[k]
            new_field[0, j] = bc_value
            new_field[ny - 1, j] = bc_value
        for i in range(ny):
            new_field[i, 0] = bc_value
            new_field[i, nx - 1] = bc_value
    else:
        # ---- Neumann (no-flux): full lines, mirror ghosts -> end diag 1+h ----
        for i in prange(ny):
            a = np.full(nx, -h)
            b = np.full(nx, 1.0 + r)
            c = np.full(nx, -h)
            b[0] = 1.0 + h
            b[nx - 1] = 1.0 + h
            d = np.empty(nx)
            for j in range(nx):
                up = field[i - 1, j] if i > 0 else field[i, j]
                dn = field[i + 1, j] if i < ny - 1 else field[i, j]
                d[j] = field[i, j] + h * (up - 2.0 * field[i, j] + dn)
            sol = _thomas(a, b, c, d)
            for j in range(nx):
                ustar[i, j] = sol[j]
        for j in prange(nx):
            a = np.full(ny, -h)
            b = np.full(ny, 1.0 + r)
            c = np.full(ny, -h)
            b[0] = 1.0 + h
            b[ny - 1] = 1.0 + h
            d = np.empty(ny)
            for i in range(ny):
                lf = ustar[i, j - 1] if j > 0 else ustar[i, j]
                rt = ustar[i, j + 1] if j < nx - 1 else ustar[i, j]
                d[i] = ustar[i, j] + h * (lf - 2.0 * ustar[i, j] + rt)
            sol = _thomas(a, b, c, d)
            for i in range(ny):
                new_field[i, j] = sol[i]

    for i in prange(ny):
        for j in range(nx):
            if new_field[i, j] < 0.0:
                new_field[i, j] = 0.0
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
