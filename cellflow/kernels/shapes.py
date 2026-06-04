"""Cell-shape mechanics: per-cell contact (Cauchy) stress driving an
area-conserving elliptical deformation.

Each cell carries a deviatoric strain (exx, exy) that evolves by a linear
viscoelastic law toward chi * (deviatoric contact stress), relaxing back to
round when the load is removed (see CellSimulation._update_shapes). The shape
is genuine dynamical state, not a render-time readout, but the *mechanics*
remain circular (cheap) -- the ellipse is the cell's elastic response to the
pressure it feels, not a new contact model.

Stress (per cell i, repulsive contacts):
    sigma_ab = (1/A_i) sum_j  f_a r_b ,  f = force on i from j, r = x_j - x_i,
deviatoric part s = [ (sigma_xx - sigma_yy)/2 , sigma_xy ].
Compression along x (sigma_xx < 0) gives s_xx < 0 -> strain exx < 0 -> the cell
shortens along x and (area-conserving) elongates along y. Isotropic pressure
has zero deviatoric part -> no elongation, as it must.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def contact_stress_celllist_numba(positions, radii, repulsion_strength,
                                  order, bin_start, nbx, bin_size):
    """Per-cell deviatoric contact stress from repulsive overlaps (cell-list).

    Returns (n, 2): [s_xx_dev, s_xy] = [(sigma_xx - sigma_yy)/2, sigma_xy],
    each already divided by the cell area. Same force law as
    calculate_repulsion_forces_numba.
    """
    n = positions.shape[0]
    out = np.zeros((n, 2))
    for i in prange(n):
        bx = int(positions[i, 0] / bin_size)
        by = int(positions[i, 1] / bin_size)
        if bx < 0: bx = 0
        elif bx >= nbx: bx = nbx - 1
        if by < 0: by = 0
        elif by >= nbx: by = nbx - 1
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
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
                    if 0.0 < dist < touch:
                        overlap = touch - dist
                        fmag = repulsion_strength * np.exp(3.0 * overlap / touch)
                        invd = 1.0 / dist
                        fx = -fmag * dx_ * invd      # force on i (away from j)
                        fy = -fmag * dy_ * invd
                        sxx += fx * dx_
                        syy += fy * dy_
                        sxy += 0.5 * (fx * dy_ + fy * dx_)
        area = np.pi * radii[i] * radii[i]
        out[i, 0] = 0.5 * (sxx - syy) / area
        out[i, 1] = sxy / area
    return out
