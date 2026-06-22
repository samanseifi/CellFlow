"""Linked-cell (cell-list) neighbor search to make the pairwise force loops
near-linear instead of O(N^2).

Cells are binned into a uniform grid whose bin size is >= the largest
interaction range, so every interacting pair lands in the same or an adjacent
bin. Each cell then only checks the 3x3 block of bins around it. When the bin
size covers the true cutoff, the resulting forces are identical to the
brute-force O(N^2) kernels (verified in tests).

The build uses a counting sort (no Python-side dynamic lists): ``bin_start``
holds the prefix-sum offsets and ``order`` lists cell indices grouped by bin.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def build_cell_list_numba(positions, physical_size, bin_size):
    """Bin cells into a uniform grid via counting sort.

    Returns
    -------
    order : (N,) int array of cell indices grouped by bin.
    bin_start : (nbins + 1,) int array of per-bin start offsets into ``order``.
    nbx : int, number of bins per axis.
    """
    n = positions.shape[0]
    nbx = int(physical_size / bin_size) + 1
    nbins = nbx * nbx

    counts = np.zeros(nbins + 1, dtype=np.int64)
    bin_of = np.empty(n, dtype=np.int64)
    for k in range(n):
        bx = int(positions[k, 0] / bin_size)
        by = int(positions[k, 1] / bin_size)
        if bx < 0: bx = 0
        elif bx >= nbx: bx = nbx - 1
        if by < 0: by = 0
        elif by >= nbx: by = nbx - 1
        b = by * nbx + bx
        bin_of[k] = b
        counts[b + 1] += 1

    # prefix sum -> bin_start
    for b in range(nbins):
        counts[b + 1] += counts[b]
    bin_start = counts

    order = np.empty(n, dtype=np.int64)
    cursor = bin_start[:nbins].copy()
    for k in range(n):
        b = bin_of[k]
        order[cursor[b]] = k
        cursor[b] += 1

    return order, bin_start, nbx


@njit(parallel=True, cache=True)
def repulsion_forces_celllist_numba(positions, radii, repulsion_strength,
                                    order, bin_start, nbx, bin_size):
    """Cell-list version of calculate_repulsion_forces_numba."""
    n = positions.shape[0]
    forces = np.zeros((n, 2))
    for i in prange(n):
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
                    dist = np.sqrt(dx_**2 + dy_**2)
                    touch = radii[i] + radii[j]
                    if 0.0 < dist < touch:
                        overlap = touch - dist
                        force_mag = repulsion_strength * np.exp(3.0 * overlap / touch)
                        inv_dist = 1.0 / dist
                        forces[i, 0] -= force_mag * dx_ * inv_dist
                        forces[i, 1] -= force_mag * dy_ * inv_dist
    return forces


@njit(parallel=True, cache=True)
def contact_pressure_celllist_numba(positions, radii, repulsion_strength,
                                    order, bin_start, nbx, bin_size):
    """Per-cell compressive contact pressure from repulsive overlaps.

    A scalar (virial) pressure for each cell,
        P_i = 1/(2 A_i) * sum_j  f_ij . r_ij,   A_i = pi R_i^2,
    where the sum runs over overlapping neighbours and, for the exponential
    repulsion, f_ij . r_ij = |f_ij| * d_ij >= 0 (compression). Unlike the net
    force vector, this does NOT cancel under isotropic squeezing, so it is a
    monotone measure of how mechanically crowded a cell is -- used to gate
    proliferation (contact inhibition / homeostatic pressure)."""
    n = positions.shape[0]
    pressure = np.zeros(n)
    for i in prange(n):
        bx = int(positions[i, 0] / bin_size)
        by = int(positions[i, 1] / bin_size)
        if bx < 0: bx = 0
        elif bx >= nbx: bx = nbx - 1
        if by < 0: by = 0
        elif by >= nbx: by = nbx - 1
        acc = 0.0
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
                    dist = np.sqrt(dx_**2 + dy_**2)
                    touch = radii[i] + radii[j]
                    if 0.0 < dist < touch:
                        overlap = touch - dist
                        force_mag = repulsion_strength * np.exp(3.0 * overlap / touch)
                        acc += force_mag * dist          # f_ij . r_ij
        area = np.pi * radii[i] * radii[i]
        pressure[i] = acc / (2.0 * area) if area > 0.0 else 0.0
    return pressure


@njit(parallel=True, cache=True)
def adhesion_forces_celllist_numba(positions, radii, adhesion_strength,
                                   adhesion_cutoff_factor,
                                   order, bin_start, nbx, bin_size):
    """Cell-list version of calculate_adhesion_forces_numba."""
    n = positions.shape[0]
    forces = np.zeros((n, 2))
    for i in prange(n):
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
                    distance = np.sqrt(dx_**2 + dy_**2)
                    touching_dist = radii[i] + radii[j]
                    cutoff_dist = touching_dist * adhesion_cutoff_factor
                    if touching_dist < distance < cutoff_dist:
                        force_magnitude = adhesion_strength * (distance - touching_dist)
                        inv_dist = 1.0 / distance
                        forces[i, 0] += force_magnitude * dx_ * inv_dist
                        forces[i, 1] += force_magnitude * dy_ * inv_dist
    return forces


@njit(parallel=True, cache=True)
def differential_adhesion_celllist_numba(positions, radii, types, adhesion_matrix,
                                         adhesion_cutoff_factor,
                                         order, bin_start, nbx, bin_size):
    """Cell-list version of calculate_differential_adhesion_forces_numba."""
    n = positions.shape[0]
    forces = np.zeros((n, 2))
    for i in prange(n):
        ti = types[i]
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
                    distance = np.sqrt(dx_**2 + dy_**2)
                    touching_dist = radii[i] + radii[j]
                    cutoff_dist = touching_dist * adhesion_cutoff_factor
                    if touching_dist < distance < cutoff_dist:
                        strength = adhesion_matrix[ti, types[j]]
                        force_magnitude = strength * (distance - touching_dist)
                        inv_dist = 1.0 / distance
                        forces[i, 0] += force_magnitude * dx_ * inv_dist
                        forces[i, 1] += force_magnitude * dy_ * inv_dist
    return forces


@njit(parallel=True, cache=True)
def resolve_overlaps_celllist_numba(positions, radii, order, bin_start, nbx, bin_size):
    """Parallel (Jacobi) overlap resolution using the cell list.

    Replaces the O(N^2) sequential resolve_overlaps_numba. Each cell reads the
    CURRENT positions (read-only) and accumulates the half-overlap push from each
    overlapping neighbour into its own displacement row; all displacements are
    then applied at once. Because every pair contributes equal-and-opposite
    pushes, the center of mass is preserved exactly. For an isolated pair this
    resolves the overlap in a single sweep (identical to the sequential kernel);
    in dense packings it converges over the configured number of sweeps.

    The exactly-coincident case is separated along a deterministic golden-angle
    direction (per cell index), preserving reproducibility.
    """
    n = positions.shape[0]
    disp = np.zeros((n, 2))
    for i in prange(n):
        xi = positions[i, 0]
        yi = positions[i, 1]
        ri = radii[i]
        bx = int(xi / bin_size)
        by = int(yi / bin_size)
        if bx < 0: bx = 0
        elif bx >= nbx: bx = nbx - 1
        if by < 0: by = 0
        elif by >= nbx: by = nbx - 1
        dxacc = 0.0
        dyacc = 0.0
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
                    dx_ = positions[j, 0] - xi
                    dy_ = positions[j, 1] - yi
                    dist = np.sqrt(dx_ * dx_ + dy_ * dy_)
                    touch = ri + radii[j]
                    if dist < 1e-12:
                        ang = i * 2.39996322972865332      # golden angle
                        dxacc -= 0.5 * touch * np.cos(ang)
                        dyacc -= 0.5 * touch * np.sin(ang)
                    elif dist < touch:
                        f = 0.5 * (touch - dist) / dist     # half the overlap
                        dxacc -= f * dx_                      # push i away from j
                        dyacc -= f * dy_
        disp[i, 0] = dxacc
        disp[i, 1] = dyacc
    for i in prange(n):
        positions[i, 0] += disp[i, 0]
        positions[i, 1] += disp[i, 1]
