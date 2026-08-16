"""Measure the EMERGENT surface tension from the stress tensor (Kirkwood-Buff).

The shape route to surface tension -- watch a square round -- confounds two very
different things. A square that does not round may have no surface tension, or it
may have one and be unable to reach the lower-energy state. This study spent a
long time unable to tell those apart, and #40 showed the second is what actually
happens: the pack jams in a local minimum with a driving force of 13 bonds in
4646.

The mechanical route separates them. For a flat interface with its normal along
x, the surface tension is the integrated anisotropy of the stress tensor across
it (Kirkwood & Buff 1949):

    gamma = integral [ p_N(x) - p_T(x) ] dx = integral [ sigma_xx - sigma_yy ] dx

This is a THERMODYNAMIC quantity read off the current configuration. It does not
require the system to move, so it is immune to jamming: a frozen pack with a
cohesive well still reports the surface tension it possesses. Comparing it with
the value inferred from shape relaxation is then a direct test of whether a null
shape result is thermodynamic or kinetic.

Stress is the pair (virial) contribution,

    sigma_ab = -(1/A) sum_{i<j} f_ij,a r_ij,b

binned along x. The kinetic/ideal-gas term is omitted: these cells have no
thermal velocities, and in an overdamped active system the momentum flux is not
a temperature. That makes this the CONFIGURATIONAL surface tension, which is what
the cohesive well contributes and what a slow shape relaxation would express.

Sign convention: gamma > 0 for a cohesive interface, i.e. one that would pull
itself flat.
"""
import numpy as np


def virial_stress_profile(positions, forces_pairwise, bins, box_length_y):
    """Bin the pair virial along x.

    Parameters
    ----------
    positions : (M, 2) array of pair MIDPOINTS
    forces_pairwise : (M, 2) array, force on i from j for each pair
    bins : (nb + 1,) bin edges along x
    box_length_y : float, transverse extent, for the per-bin area

    Returns
    -------
    sxx, syy : (nb,) arrays of stress components.
    """
    nb = len(bins) - 1
    dx = bins[1] - bins[0]
    area = dx * box_length_y
    idx = np.clip(np.digitize(positions[:, 0], bins) - 1, 0, nb - 1)
    sxx = np.zeros(nb)
    syy = np.zeros(nb)
    np.add.at(sxx, idx, forces_pairwise[:, 0] * positions[:, 2]
              if positions.shape[1] > 2 else 0.0)
    return sxx / area, syy / area


def pair_virial(positions, radii, force_fn, cutoff):
    """All interacting pairs with their separation vectors and forces.

    ``force_fn(delta, R_reduced)`` returns the SCALAR force along the centre
    line, positive = repulsive, matching the convention of the JKR kernel.
    """
    n = len(positions)
    mids, rij, fij = [], [], []
    for i in range(n):
        d = positions[i + 1:] - positions[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        touch = radii[i] + radii[i + 1:]
        m = (dist < cutoff * touch) & (dist > 1e-12)
        if not m.any():
            continue
        dd = d[m]
        dist_m = dist[m]
        touch_m = touch[m]
        delta = touch_m - dist_m
        R_red = radii[i] * radii[i + 1:][m] / touch_m
        fmag = np.array([force_fn(dl, R) for dl, R in zip(delta, R_red)])
        unit = dd / dist_m[:, None]
        mids.append(positions[i] + 0.5 * dd)
        rij.append(dd)
        # force on i from j: repulsive (fmag > 0) pushes i AWAY from j
        fij.append(-fmag[:, None] * unit)
    if not mids:
        return (np.zeros((0, 2)),) * 3
    return np.vstack(mids), np.vstack(rij), np.vstack(fij)


def surface_tension_kirkwood_buff(positions, radii, force_fn, cutoff,
                                  n_bins=120, x_range=None):
    """Configurational surface tension of a slab with interfaces normal to x.

    Returns
    -------
    gamma : float
        Surface tension PER INTERFACE (a slab has two, so the integral is
        halved).
    profile : dict with the binned stress components and bin centres.
    """
    mids, rij, fij = pair_virial(positions, radii, force_fn, cutoff)
    if len(mids) == 0:
        return 0.0, {}

    if x_range is None:
        x_range = (positions[:, 0].min() - 2.0, positions[:, 0].max() + 2.0)
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    dx = bins[1] - bins[0]
    Ly = positions[:, 1].max() - positions[:, 1].min()
    area = dx * Ly

    idx = np.clip(np.digitize(mids[:, 0], bins) - 1, 0, n_bins - 1)
    sxx = np.zeros(n_bins)
    syy = np.zeros(n_bins)
    # sigma_ab = -(1/A) sum f_a r_b ; f here is the force on i from j and r is
    # r_j - r_i, so the product already carries the right sign for a stress.
    np.add.at(sxx, idx, -fij[:, 0] * rij[:, 0])
    np.add.at(syy, idx, -fij[:, 1] * rij[:, 1])
    sxx /= area
    syy /= area

    centres = 0.5 * (bins[:-1] + bins[1:])
    gamma = 0.5 * np.sum(sxx - syy) * dx        # two interfaces in a slab
    return float(gamma), {'x': centres, 'sxx': sxx, 'syy': syy,
                          'anisotropy': sxx - syy, 'dx': dx}


def surface_tension_from_bond_counting(n_broken_per_length, well_depth):
    """Independent estimate: gamma ~ (broken bonds per unit length) x (well depth) / 2.

    The factor of 1/2 is the usual double-counting correction -- each broken bond
    is shared between the two surfaces created. Crude, but it is derived from the
    pair potential alone and so provides a check on the virial result that shares
    none of its machinery.
    """
    return 0.5 * n_broken_per_length * abs(well_depth)
