"""Local friction velocity law -- the center-based-model standard.

    gamma_sub * v_i  +  sum_j gamma_cc * w_ij * (v_i - v_j)  =  F_i

Each cell's velocity responds to its OWN force through substrate drag, coupled to
neighbours by cell-cell friction. This is the velocity law used throughout the
center-based-model literature (Van Liedekerke, Palm, Jagiella & Drasdo, Comp.
Part. Mech. 2:401-444, 2015, Sect. 3.1.1), and it is an alternative to -- not a
replacement for -- the Brinkman/IBM path, which remains correct for cells in
suspension.

Why it exists here
------------------
Under the fluid path a cell's velocity is the interpolated fluid velocity at its
position, ``v_cell = u_fluid(x_cell)``. That advects every cell by one smooth
field, so two neighbours can never acquire relative velocity at the cell scale
and can never exchange places. Measured consequences (docs/giverso_replication.md):

  * a random propulsive force of 150 per cell -- three times the mean repulsive
    force -- changed the mean displacement over 500 steps from 3.154 to 3.159,
    i.e. not at all;
  * a square of cells does not relax into a disc, and adhesion = 0 is
    indistinguishable from adhesion = 50.

The cause is the Brinkman transfer function ``u_hat = f_hat/(mu k^2 + alpha)``,
which suppresses force at wavenumber k by ``1 + (k delta)^2`` -- 416x at the cell
scale for the colony settings. T1 neighbour exchange is what lets real tissue
behave as a liquid and round up, and it is exactly the mode that filter removes.

Structure of the operator
-------------------------
A = gamma_sub * I + L, where L is the weighted graph Laplacian of the contact
network. L is symmetric positive SEMI-definite (it annihilates uniform velocity),
so with gamma_sub > 0 the whole operator is symmetric positive definite and
Conjugate Gradient converges. The solve is matrix-free: only the matvec is
needed, and it costs one pass over the contact neighbours.

Galilean invariance is exact and structural: L kills a uniform velocity, so
adding a constant to every v changes ``A v`` only through the ``gamma_sub`` term
-- which is the physically correct statement that substrate drag, not cell-cell
friction, is what breaks translation invariance.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _contact_weight(dist, touch, cutoff):
    """Friction weight for a pair, a proxy for shared contact area.

    1 at full overlap, tapering linearly to 0 at ``cutoff``. Continuous
    everywhere, so the velocity field has no jump as a contact forms or breaks
    -- unlike the repulsion law, whose discontinuity at contact is issue #33.
    """
    if dist >= cutoff:
        return 0.0
    if dist <= touch:
        return 1.0
    return (cutoff - dist) / (cutoff - touch)


@njit(parallel=True, cache=True)
def friction_matvec_numba(v, positions, radii, gamma_sub, gamma_cc,
                          cutoff_factor, order, bin_start, nbx, bin_size):
    """Apply A = gamma_sub*I + L to ``v``; L the contact-network Laplacian."""
    n = positions.shape[0]
    out = np.zeros((n, 2))
    for i in prange(n):
        acc_x = gamma_sub * v[i, 0]
        acc_y = gamma_sub * v[i, 1]
        bx = int(positions[i, 0] / bin_size)
        by = int(positions[i, 1] / bin_size)
        if bx < 0: bx = 0
        elif bx >= nbx: bx = nbx - 1
        if by < 0: by = 0
        elif by >= nbx: by = nbx - 1
        for dby in range(-1, 2):
            ny = by + dby
            if ny < 0 or ny >= nbx:
                continue
            for dbx in range(-1, 2):
                nx = bx + dbx
                if nx < 0 or nx >= nbx:
                    continue
                b = ny * nbx + nx
                for s in range(bin_start[b], bin_start[b + 1]):
                    j = order[s]
                    if j == i:
                        continue
                    dx_ = positions[j, 0] - positions[i, 0]
                    dy_ = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx_ * dx_ + dy_ * dy_)
                    touch = radii[i] + radii[j]
                    w = _contact_weight(dist, touch, touch * cutoff_factor)
                    if w > 0.0:
                        g = gamma_cc * w
                        acc_x += g * (v[i, 0] - v[j, 0])
                        acc_y += g * (v[i, 1] - v[j, 1])
        out[i, 0] = acc_x
        out[i, 1] = acc_y
    return out


def solve_friction_velocities(positions, radii, forces, gamma_sub, gamma_cc,
                              cutoff_factor=1.0, tol=1e-8, max_iter=200,
                              cell_list=None, physical_size=None, v0=None):
    """Solve ``A v = F`` for the cell velocities by Conjugate Gradient.

    Parameters
    ----------
    positions, radii, forces : (N, 2), (N,), (N, 2) arrays
    gamma_sub : float
        Substrate drag, > 0. Sets the isolated-cell mobility: a cell with no
        neighbours moves at ``v = F / gamma_sub`` exactly.
    gamma_cc : float
        Cell-cell friction coefficient (>= 0). ``0`` decouples the cells
        entirely, giving pure local mobility.
    cutoff_factor : float
        Friction range as a multiple of the touching distance. ``1.0`` means
        only genuinely overlapping cells rub.
    cell_list : tuple, optional
        Prebuilt ``(order, bin_start, nbx, bin_size)`` to avoid rebuilding.
    v0 : (N, 2) array, optional
        Initial guess; the previous step's velocities make a good one.

    Returns
    -------
    v : (N, 2) array
    info : dict with ``iterations`` and ``residual``.
    """
    from .neighbors import build_cell_list_numba

    if gamma_sub <= 0.0:
        raise ValueError("gamma_sub must be positive (it makes A definite)")
    if gamma_cc < 0.0:
        raise ValueError("gamma_cc must be non-negative")

    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 2)), {'iterations': 0, 'residual': 0.0}

    if cell_list is None:
        bin_size = 2.0 * float(radii.max()) * max(cutoff_factor, 1.0)
        if physical_size is None:
            physical_size = float(positions.max()) + bin_size
        order, bin_start, nbx = build_cell_list_numba(
            positions, physical_size, bin_size)
    else:
        order, bin_start, nbx, bin_size = cell_list

    def A(x):
        return friction_matvec_numba(x, positions, radii, gamma_sub, gamma_cc,
                                     cutoff_factor, order, bin_start, nbx,
                                     bin_size)

    # Jacobi preconditioner: the diagonal is gamma_sub + sum_j gamma_cc w_ij,
    # recovered by applying A to a unit vector per component would be wrong
    # (that includes off-diagonal terms), so take the row sums of the coupling
    # via A(1) - which for uniform v gives exactly gamma_sub * 1.
    # Instead use the cheap and adequate constant preconditioner gamma_sub.
    b = np.ascontiguousarray(forces, dtype=np.float64)
    v = np.zeros((n, 2)) if v0 is None else np.array(v0, dtype=np.float64)

    r = b - A(v)
    z = r / gamma_sub
    p = z.copy()
    rz = float(np.sum(r * z))
    bnorm = float(np.sqrt(np.sum(b * b)))
    if bnorm == 0.0:
        return np.zeros((n, 2)), {'iterations': 0, 'residual': 0.0}

    it = 0
    res = float(np.sqrt(np.sum(r * r))) / bnorm
    for it in range(1, max_iter + 1):
        if res <= tol:
            it -= 1
            break
        Ap = A(p)
        pAp = float(np.sum(p * Ap))
        if pAp <= 0.0:                      # cannot happen for SPD; guard anyway
            break
        alpha = rz / pAp
        v += alpha * p
        r -= alpha * Ap
        res = float(np.sqrt(np.sum(r * r))) / bnorm
        z = r / gamma_sub
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new

    return v, {'iterations': it, 'residual': res}
