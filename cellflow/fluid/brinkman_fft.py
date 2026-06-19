"""FFT-based incompressible Brinkman (and Stokes) flow solver on a periodic grid.

Solves, for a force density ``f`` on a doubly-periodic grid:

    -mu * laplacian(u) + alpha * u + grad(p) = f,   div(u) = 0

In Fourier space, for each wavevector k != 0:

    u_hat(k) = P(k) f_hat(k) / (mu |k|^2 + alpha),   P(k) = I - k k^T / |k|^2

where P is the Leray (divergence-free) projection. The mean mode (k = 0) is

    u_hat(0) = f_hat(0) / alpha          (finite when alpha > 0)

The substrate-drag term ``alpha`` models dish-bottom / thin-film friction and
introduces a screening length ``delta = sqrt(mu / alpha)`` beyond which
hydrodynamic interactions decay exponentially. This regularizes the 2D Stokes
paradox (the free-space 2D Stokeslet velocity grows like ln(r); here it
decays). Setting ``alpha = 0`` recovers pure periodic Stokes flow, valid only
for force-free inputs (zero net force); the mean mode is then set to zero.

Cost is O(M log M) for M grid points (two FFTs per component).
"""
import numpy as np


def alpha_from_screening_length(mu, screening_length):
    """Convert a screening length ``delta`` to the Brinkman drag ``alpha``.

    delta = sqrt(mu / alpha)  =>  alpha = mu / delta**2
    """
    if screening_length <= 0.0:
        raise ValueError("screening_length must be positive")
    return mu / (screening_length ** 2)


def _wavenumbers(shape, dx):
    """Angular wavenumber grids (kx, ky) and |k|^2 for an (ny, nx) field.

    For even grid sizes the Nyquist wavenumber is zeroed. On a real field the
    Nyquist mode cannot carry a consistent first derivative (its sign is
    ambiguous), so keeping it would break the divergence-free projection. This
    is standard practice for pseudo-spectral first-derivative / Leray operators.
    """
    ny, nx = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)   # length nx
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)   # length ny
    if nx % 2 == 0:
        kx[nx // 2] = 0.0
    if ny % 2 == 0:
        ky[ny // 2] = 0.0
    KX, KY = np.meshgrid(kx, ky)                   # (ny, nx)
    K2 = KX**2 + KY**2
    return KX, KY, K2


def solve_velocity(force_density, mu, dx, alpha=0.0, screening_length=None):
    """Solve incompressible Brinkman/Stokes flow for a periodic force density.

    Parameters
    ----------
    force_density : (ny, nx, 2) array
        Body-force density on the grid (component 0 = x, component 1 = y).
    mu : float
        Dynamic viscosity (must be > 0).
    dx : float
        Grid spacing (uniform, square cells).
    alpha : float, optional
        Brinkman substrate-drag coefficient (>= 0). ``alpha = 0`` is pure Stokes.
    screening_length : float, optional
        If given, overrides ``alpha`` via ``alpha = mu / screening_length**2``.

    Returns
    -------
    u : (ny, nx, 2) real array
        Divergence-free velocity field.
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if screening_length is not None:
        alpha = alpha_from_screening_length(mu, screening_length)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    fx = force_density[:, :, 0]
    fy = force_density[:, :, 1]

    fx_hat = np.fft.fft2(fx)
    fy_hat = np.fft.fft2(fy)

    KX, KY, K2 = _wavenumbers(fx.shape, dx)

    # Leray projection: f_perp = f - k (k . f) / |k|^2
    k_dot_f = KX * fx_hat + KY * fy_hat
    K2_safe = np.where(K2 == 0.0, 1.0, K2)  # avoid 0/0 at the mean mode
    fx_perp = fx_hat - KX * k_dot_f / K2_safe
    fy_perp = fy_hat - KY * k_dot_f / K2_safe

    denom = mu * K2 + alpha
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    ux_hat = fx_perp / denom_safe
    uy_hat = fy_perp / denom_safe

    # Mean mode (k = 0): u_hat(0) = f_hat(0) / alpha for Brinkman; 0 for pure Stokes.
    if alpha > 0.0:
        ux_hat[0, 0] = fx_hat[0, 0] / alpha
        uy_hat[0, 0] = fy_hat[0, 0] / alpha
    else:
        ux_hat[0, 0] = 0.0
        uy_hat[0, 0] = 0.0

    ux = np.fft.ifft2(ux_hat).real
    uy = np.fft.ifft2(uy_hat).real

    u = np.empty_like(force_density)
    u[:, :, 0] = ux
    u[:, :, 1] = uy
    return u


def _mirror_extend(a, axis, parity):
    """Half-sample-symmetric mirror extension along ``axis``, doubling its length.

    ``parity = +1`` is an even reflection (the extension underlying the DCT-II),
    ``parity = -1`` an odd reflection (underlying the DST-II). For cell-centered
    data ``[a_0, ..., a_{N-1}]`` the result is ``[a_0, ..., a_{N-1}, +/-a_{N-1},
    ..., +/-a_0]``, placing mirror (symmetry) planes at both ends of the domain.
    """
    rev = np.flip(a, axis=axis)
    if parity < 0:
        rev = -rev
    return np.concatenate([a, rev], axis=axis)


def solve_velocity_freeslip_box(force_density, mu, dx, alpha=0.0,
                                screening_length=None):
    """Incompressible Brinkman/Stokes flow in a box with FREE-SLIP walls.

    Solves the same equations as :func:`solve_velocity` on the domain
    ``[0, L]^2`` but with free-slip (no-penetration, stress-free) walls on all
    four sides:

        u . n = 0   and   tangential stress = 0   on each wall.

    A free-slip wall is a plane of mirror symmetry, so the box solution is the
    restriction of a periodic solution on the doubled domain ``[0, 2L]^2`` whose
    force has the symmetry

        f_x : odd across the x-walls, even across the y-walls,
        f_y : even across the x-walls, odd across the y-walls.

    We build that mirror-extended force and reuse the verified periodic solver.
    The Brinkman multiplier ``1/(mu|k|^2 + alpha)`` and the Leray projection both
    map this symmetry subspace to itself, so the solved velocity inherits the
    symmetry (``u_x`` odd in x / even in y, ``u_y`` even in x / odd in y), which
    is exactly free slip on the box walls, and it is divergence-free. The
    original quadrant is then cropped out. This is the standard equivalence
    between a DST/DCT solve and an FFT on mirror-extended data.

    Cost is 4x the periodic solve (a 2N x 2N grid) -- still O(M log M).
    Parameters match :func:`solve_velocity`; ``force_density`` is (ny, nx, 2).
    """
    ny, nx, _ = force_density.shape
    fx = force_density[:, :, 0]
    fy = force_density[:, :, 1]

    # axis 0 = y, axis 1 = x.
    # f_x: odd in x (parity -1, axis 1), even in y (parity +1, axis 0).
    fx_ext = _mirror_extend(_mirror_extend(fx, axis=1, parity=-1),
                            axis=0, parity=+1)
    # f_y: even in x (parity +1, axis 1), odd in y (parity -1, axis 0).
    fy_ext = _mirror_extend(_mirror_extend(fy, axis=1, parity=+1),
                            axis=0, parity=-1)

    f_ext = np.empty((2 * ny, 2 * nx, 2), dtype=force_density.dtype)
    f_ext[:, :, 0] = fx_ext
    f_ext[:, :, 1] = fy_ext

    u_ext = solve_velocity(f_ext, mu, dx, alpha=alpha,
                           screening_length=screening_length)
    return u_ext[:ny, :nx, :].copy()


def solve_velocity_variable_alpha(force_density, mu, dx, alpha_field,
                                  tol=1e-7, max_iter=300):
    """Incompressible Brinkman with a SPATIALLY-VARYING drag alpha(x):

        -mu*lap(u) + alpha(x) u + grad(p) = f,   div(u) = 0.

    Variable alpha is not diagonal in Fourier, so we split alpha = alpha0 + d(x)
    with alpha0 = max(alpha) and iterate (Picard), using the constant-alpha FFT
    solver A^{-1}P as the inner solve:

        u <- A0^{-1} P ( f - d(x) u ) ,   A0 = -mu*lap + alpha0.

    Because d <= 0 and |d|/(mu|k|^2 + alpha0) < 1, the iteration contracts, with
    rate ~ (1 - alpha_min/alpha_max). It is therefore efficient for MODERATE
    drag contrast (a few x); very high contrast (>~50x) converges slowly and
    would warrant a preconditioned Krylov solver (future work). Returns
    (u, iterations, residual).

    Parameters
    ----------
    alpha_field : (ny, nx) array of non-negative drag values.
    """
    alpha0 = float(np.max(alpha_field))
    if alpha0 <= 0.0:
        # uniform Stokes (alpha == 0 everywhere)
        return solve_velocity(force_density, mu, dx, alpha=0.0), 1, 0.0
    delta = (alpha_field - alpha0)[:, :, None]      # <= 0, shape (ny,nx,1)

    u = solve_velocity(force_density, mu, dx, alpha=alpha0)
    residual = 0.0
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        rhs = force_density - delta * u
        u_new = solve_velocity(rhs, mu, dx, alpha=alpha0)
        denom = np.max(np.abs(u_new)) + 1e-30
        residual = np.max(np.abs(u_new - u)) / denom
        u = u_new
        if residual < tol:
            break
    return u, iters, residual


def spectral_divergence(u, dx):
    """Return the divergence field of ``u`` computed spectrally (for tests)."""
    ux_hat = np.fft.fft2(u[:, :, 0])
    uy_hat = np.fft.fft2(u[:, :, 1])
    KX, KY, _ = _wavenumbers(u[:, :, 0].shape, dx)
    div_hat = 1j * (KX * ux_hat + KY * uy_hat)
    return np.fft.ifft2(div_hat).real
