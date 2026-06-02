"""Hydrodynamics kernels: 2D regularized Stokeslet fluid velocity, cell
mobility, and in-cell velocity damping."""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def update_fluid_velocity_numba(fluid_velocity, positions, forces, viscosity, dx,
                                 cutoff_radius=-1.0):
    """Calculates fluid velocity on the grid using the 2D regularized Stokeslet.

    The 2D regularized Stokeslet (Cortez 2001) for a point force f at x0:
        v_i(x) = (1/4*pi*mu) * [-f_i * H1 + (f . r) * r_i * H2]
    where r = x - x0, r_eps = sqrt(|r|^2 + eps^2),
        H1 = ln(r_eps) - eps^2 / r_eps^2,
        H2 = 1 / r_eps^2.

    cutoff_radius: If > 0, skip contributions from cells farther than this physical
                   distance (approximate but faster for concentrated cell clusters).
    """
    h, w, _ = fluid_velocity.shape
    n = len(positions)
    eps = 2.0 * dx      # Regularization parameter
    eps2 = eps * eps
    pref = 1.0 / (4.0 * np.pi * viscosity)  # 2D prefactor
    use_cutoff = cutoff_radius > 0.0
    cutoff2 = cutoff_radius * cutoff_radius

    for gy in prange(h):
        for gx in range(w):
            xg = gx * dx
            yg = gy * dx
            vx_total, vy_total = 0.0, 0.0

            for k in range(n):
                px, py = positions[k, 0], positions[k, 1]
                fx, fy = forces[k, 0], forces[k, 1]

                rx = xg - px
                ry = yg - py
                r2 = rx*rx + ry*ry

                if use_cutoff and r2 > cutoff2:
                    continue

                # 2D regularized Stokeslet kernel
                r_eps2 = r2 + eps2
                r_eps = np.sqrt(r_eps2)
                H1 = np.log(r_eps) - eps2 / r_eps2
                H2 = 1.0 / r_eps2
                f_dot_r = fx * rx + fy * ry

                vx_total += -fx * H1 + f_dot_r * rx * H2
                vy_total += -fy * H1 + f_dot_r * ry * H2

            fluid_velocity[gy, gx, 0] = pref * vx_total
            fluid_velocity[gy, gx, 1] = pref * vy_total

    return fluid_velocity


@njit(parallel=True, cache=True)
def update_fluid_velocity_with_dipoles_numba(fluid_velocity, positions, monopolar_forces,
                                              propulsive_forces, orientations,
                                              cell_dipole_lengths, viscosity, dx,
                                              cutoff_radius=-1.0):
    """
    Calculates fluid velocity using 2D regularized Stokeslets for monopolar forces
    and a pair of equal-and-opposite 2D regularized Stokeslets (force dipole) for
    propulsive (swimming) forces.

    cutoff_radius: If > 0, skip cells farther than this physical distance from a
                   grid point (approximate but faster for concentrated cell clusters).
    """
    h, w, _ = fluid_velocity.shape
    num_cells = len(positions)
    eps = 2.0 * dx
    eps2 = eps * eps
    pref = 1.0 / (4.0 * np.pi * viscosity)  # 2D prefactor
    use_cutoff = cutoff_radius > 0.0
    cutoff2 = cutoff_radius * cutoff_radius

    for gy in prange(h):
        for gx in range(w):
            xg = gx * dx
            yg = gy * dx
            vx_total, vy_total = 0.0, 0.0

            for k in range(num_cells):
                px_k, py_k = positions[k, 0], positions[k, 1]

                rx_k = xg - px_k
                ry_k = yg - py_k
                r2_k = rx_k * rx_k + ry_k * ry_k
                if use_cutoff and r2_k > cutoff2:
                    continue

                # --- Part 1: Monopolar forces — 2D regularized Stokeslet ---
                f_mono_x, f_mono_y = monopolar_forces[k, 0], monopolar_forces[k, 1]
                if f_mono_x != 0.0 or f_mono_y != 0.0:
                    r_eps2 = r2_k + eps2
                    r_eps = np.sqrt(r_eps2)
                    H1 = np.log(r_eps) - eps2 / r_eps2
                    H2 = 1.0 / r_eps2
                    f_dot_r = f_mono_x * rx_k + f_mono_y * ry_k

                    vx_total += -f_mono_x * H1 + f_dot_r * rx_k * H2
                    vy_total += -f_mono_y * H1 + f_dot_r * ry_k * H2

                # --- Part 2: Propulsive dipole — two opposite 2D Stokeslets ---
                f_prop_mag = np.sqrt(propulsive_forces[k, 0]**2 + propulsive_forces[k, 1]**2)
                if f_prop_mag > 1e-9:
                    ux, uy = orientations[k, 0], orientations[k, 1]
                    L = cell_dipole_lengths[k]

                    # Front pole (+F)
                    rx_f = xg - (px_k + 0.5 * L * ux)
                    ry_f = yg - (py_k + 0.5 * L * uy)
                    r2_f = rx_f * rx_f + ry_f * ry_f
                    r_eps2_f = r2_f + eps2
                    r_eps_f = np.sqrt(r_eps2_f)
                    H1_f = np.log(r_eps_f) - eps2 / r_eps2_f
                    H2_f = 1.0 / r_eps2_f
                    ffx = f_prop_mag * ux
                    ffy = f_prop_mag * uy
                    f_dot_r_f = ffx * rx_f + ffy * ry_f
                    vx_total += -ffx * H1_f + f_dot_r_f * rx_f * H2_f
                    vy_total += -ffy * H1_f + f_dot_r_f * ry_f * H2_f

                    # Back pole (-F)
                    rx_b = xg - (px_k - 0.5 * L * ux)
                    ry_b = yg - (py_k - 0.5 * L * uy)
                    r2_b = rx_b * rx_b + ry_b * ry_b
                    r_eps2_b = r2_b + eps2
                    r_eps_b = np.sqrt(r_eps2_b)
                    H1_b = np.log(r_eps_b) - eps2 / r_eps2_b
                    H2_b = 1.0 / r_eps2_b
                    f_dot_r_b = -ffx * rx_b + -ffy * ry_b
                    vx_total += ffx * H1_b + f_dot_r_b * rx_b * H2_b
                    vy_total += ffy * H1_b + f_dot_r_b * ry_b * H2_b

            fluid_velocity[gy, gx, 0] = pref * vx_total
            fluid_velocity[gy, gx, 1] = pref * vy_total

    return fluid_velocity


@njit(parallel=True, cache=True)
def compute_cell_velocities_numba(positions, forces, radii, viscosity, dx):
    """Compute cell velocities using the 2D regularized Stokeslet mobility relation.

    For each cell k:
        v_k = F_k / (6*pi*mu*R_k)                     [self-mobility via Stokes drag]
            + sum_{j != k} G_2D(x_k, x_j) . F_j       [hydrodynamic interactions]

    Self-mobility uses the standard Stokes drag law rather than the regularized
    kernel so that the effective hydrodynamic radius equals the actual cell radius
    (independent of the grid regularization parameter eps).
    """
    n = len(positions)
    velocities = np.zeros((n, 2))
    eps = 2.0 * dx
    eps2 = eps * eps
    pref = 1.0 / (4.0 * np.pi * viscosity)

    for i in prange(n):
        vx_hydro, vy_hydro = 0.0, 0.0
        for j in range(n):
            if j == i:
                continue
            fx, fy = forces[j, 0], forces[j, 1]
            rx = positions[i, 0] - positions[j, 0]
            ry = positions[i, 1] - positions[j, 1]
            r2 = rx * rx + ry * ry

            r_eps2 = r2 + eps2
            r_eps = np.sqrt(r_eps2)
            H1 = np.log(r_eps) - eps2 / r_eps2
            H2 = 1.0 / r_eps2
            f_dot_r = fx * rx + fy * ry

            vx_hydro += -fx * H1 + f_dot_r * rx * H2
            vy_hydro += -fy * H1 + f_dot_r * ry * H2

        # Self-mobility: Stokes drag on a sphere of radius R
        drag_inv = 1.0 / (6.0 * np.pi * viscosity * radii[i])
        velocities[i, 0] = forces[i, 0] * drag_inv + pref * vx_hydro
        velocities[i, 1] = forces[i, 1] * drag_inv + pref * vy_hydro

    return velocities
