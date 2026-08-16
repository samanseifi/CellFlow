"""JKR adhesive contact mechanics for soft spheres (issue #33).

Johnson-Kendall-Roberts contact replaces the model's original pair law, which
had no derivation and a discontinuity at contact: repulsion switched on at the
touching distance with magnitude ``k_rep`` (35 at the study settings) while
adhesion was a weak linear spring acting only OUTSIDE contact (mean 0.11). The
result was a cohesive well of 0.75 against a contact repulsion of 35, which is
why the measured surface tension sat below the smallest value in Giverso et
al.'s own figure. See docs/giverso_replication.md.

The JKR relations, parameterised by contact radius ``a``:

    F(a)     = 4 E* a^3 / (3 R)  -  sqrt(8 pi w E* a^3)
    delta(a) = a^2 / R           -  sqrt(2 pi w a / E*)

with ``R`` the reduced radius R_i R_j/(R_i+R_j), ``E*`` the reduced modulus, and
``w`` the work of adhesion (energy per unit area) -- the quantity that becomes a
surface tension when integrated around a cell.

Two features matter here and neither exists in the original law:

* **An adhesive neck.** delta(a) is non-monotonic: it has a minimum at
  a_min = (cR/4)^(2/3) with c = sqrt(2 pi w / E*). Contact therefore persists to
  NEGATIVE overlap, out to delta_c = delta(a_min), and the force there is the
  pull-off force F_c = -(3/2) pi w R. Cells resist being pulled apart.
* **Hysteresis.** Because two branches of a(delta) exist for delta > delta_c,
  the path taken depends on history: an approaching pair jumps into contact,
  a separating pair holds its neck until delta_c and then snaps. Real cell-cell
  contacts do this -- bond rupture and membrane tethers are not reversible --
  and it is what lets a tissue store the energy that a surface tension is made
  of.

The physical branch is the LARGER root, a >= a_min; the smaller one is unstable.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _a_min(R, c):
    """Contact radius at the pull-off point, where d(delta)/da = 0."""
    return (c * R / 4.0) ** (2.0 / 3.0)


@njit(cache=True, inline='always')
def _delta_of_a(a, R, c):
    return a * a / R - c * np.sqrt(a)


@njit(cache=True, inline='always')
def _solve_contact_radius(delta, R, c, a_guess):
    """Invert delta(a) on the stable branch a >= a_min, by safeguarded Newton.

    Returns 0.0 when delta is below the pull-off separation, i.e. the contact
    has snapped and there is no neck left.
    """
    amin = _a_min(R, c)
    dmin = _delta_of_a(amin, R, c)
    if delta < dmin:
        return 0.0
    # bracket [amin, hi] with delta(hi) >= delta
    hi = amin if amin > a_guess else a_guess
    if hi < amin:
        hi = amin
    for _ in range(60):
        if _delta_of_a(hi, R, c) >= delta:
            break
        hi *= 2.0
    lo = amin
    a = 0.5 * (lo + hi)
    for _ in range(60):
        f = _delta_of_a(a, R, c) - delta
        if f > 0.0:
            hi = a
        else:
            lo = a
        # Newton step, rejected if it leaves the bracket
        df = 2.0 * a / R - 0.5 * c / np.sqrt(a)
        if df > 1e-30:
            an = a - f / df
            if lo < an < hi:
                a = an
            else:
                a = 0.5 * (lo + hi)
        else:
            a = 0.5 * (lo + hi)
        if hi - lo < 1e-12 * (1.0 + a):
            break
    return a


@njit(cache=True, inline='always')
def _force_of_a(a, R, E_star, w):
    """JKR force; positive = repulsive, negative = adhesive."""
    return (4.0 * E_star * a ** 3) / (3.0 * R) - np.sqrt(
        8.0 * np.pi * w * E_star * a ** 3)


@njit(cache=True)
def jkr_pair_force(delta, R, E_star, w):
    """Scalar JKR force at overlap ``delta`` (negative delta = separation)."""
    c = np.sqrt(2.0 * np.pi * w / E_star)
    a = _solve_contact_radius(delta, R, c, np.sqrt(abs(delta) * R) + 1e-12)
    if a <= 0.0:
        return 0.0
    return _force_of_a(a, R, E_star, w)


@njit(cache=True)
def jkr_pull_off_force(R, w):
    """Analytic pull-off force, F_c = -(3/2) pi w R."""
    return -1.5 * np.pi * w * R


@njit(parallel=True, cache=True)
def jkr_forces_celllist_numba(positions, radii, E_star, work_adhesion,
                              order, bin_start, nbx, bin_size):
    """Pairwise JKR forces over the contact neighbour list.

    Acts along the centre line, like the law it replaces. Separated pairs beyond
    the pull-off point contribute nothing, so the neighbour cutoff only has to
    cover the neck, not an arbitrary adhesion range.
    """
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
                    dist = np.sqrt(dx_ * dx_ + dy_ * dy_)
                    if dist < 1e-12:
                        continue
                    touch = radii[i] + radii[j]
                    delta = touch - dist
                    R = radii[i] * radii[j] / touch      # reduced radius
                    c = np.sqrt(2.0 * np.pi * work_adhesion / E_star)
                    if delta < _delta_of_a(_a_min(R, c), R, c):
                        continue                          # neck has snapped
                    a = _solve_contact_radius(
                        delta, R, c, np.sqrt(abs(delta) * R) + 1e-12)
                    if a <= 0.0:
                        continue
                    fmag = _force_of_a(a, R, E_star, work_adhesion)
                    # positive fmag = repulsive = push i away from j
                    inv = 1.0 / dist
                    forces[i, 0] -= fmag * dx_ * inv
                    forces[i, 1] -= fmag * dy_ * inv
    return forces


@njit(cache=True, inline='always')
def _stiffness_of_a(a, R, E_star, w, c):
    """dF/d(delta) at contact radius ``a``, via the chain rule through ``a``.

    dF/da   = 4 E* a^2 / R  -  (3/2) sqrt(8 pi w E* a)
    dd/da   = 2a/R          -  c / (2 sqrt(a))

    dd/da vanishes exactly at the snap point, so the contact stiffness DIVERGES
    there. That is physical -- it is the displacement-control instability -- but
    it means a stiffness-based timestep bound must be taken over the contacts
    that actually exist, not over the whole branch.
    """
    dF = 4.0 * E_star * a * a / R - 1.5 * np.sqrt(8.0 * np.pi * w * E_star * a)
    dd = 2.0 * a / R - 0.5 * c / np.sqrt(a)
    if abs(dd) < 1e-12:
        return np.inf
    return dF / dd


@njit(parallel=True, cache=True)
def jkr_max_stiffness_numba(positions, radii, E_star, work_adhesion,
                            order, bin_start, nbx, bin_size):
    """Largest contact stiffness |dF/d(delta)| over the pairs currently in contact.

    This is what sets the stable timestep: with an overdamped velocity law
    v = F/gamma, a contact of stiffness k relaxes at rate k/gamma, and explicit
    Euler needs dt < 2 gamma / k.
    """
    n = positions.shape[0]
    kmax = np.zeros(n)
    c = np.sqrt(2.0 * np.pi * work_adhesion / E_star)
    for i in prange(n):
        bx = int(positions[i, 0] / bin_size)
        by = int(positions[i, 1] / bin_size)
        if bx < 0: bx = 0
        elif bx >= nbx: bx = nbx - 1
        if by < 0: by = 0
        elif by >= nbx: by = nbx - 1
        best = 0.0
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
                    if dist < 1e-12:
                        continue
                    touch = radii[i] + radii[j]
                    delta = touch - dist
                    R = radii[i] * radii[j] / touch
                    cc = np.sqrt(2.0 * np.pi * work_adhesion / E_star)
                    if delta < _delta_of_a(_a_min(R, cc), R, cc):
                        continue
                    a = _solve_contact_radius(
                        delta, R, cc, np.sqrt(abs(delta) * R) + 1e-12)
                    if a <= 0.0:
                        continue
                    k = abs(_stiffness_of_a(a, R, E_star, work_adhesion, cc))
                    if k > best:
                        best = k
        kmax[i] = best
    return kmax


def jkr_stable_dt(positions, radii, E_star, work_adhesion, gamma_sub,
                  safety=0.25, physical_size=None):
    """Largest stable timestep for the JKR + friction pair, with a safety factor.

    Explicit Euler on an overdamped contact of stiffness k needs
    ``dt < 2 gamma / k``; ``safety`` (default 0.25) keeps a margin, since the
    stiffness rises as the pack compresses and the bound is evaluated on the
    CURRENT configuration.

    This matters because exceeding it does not blow up loudly -- the pack
    fragments quietly. Measured on the square test at dt = 0.05, the main
    cluster fell to 34% while the run still produced a plausible-looking
    circularity, which is exactly the kind of artifact this study has been
    burned by before.
    """
    from .neighbors import build_cell_list_numba

    if len(radii) == 0:
        return np.inf
    bin_size = 2.0 * float(radii.max()) * 1.5
    if physical_size is None:
        physical_size = float(positions.max()) + bin_size
    order, bin_start, nbx = build_cell_list_numba(positions, physical_size, bin_size)
    kmax = float(jkr_max_stiffness_numba(positions, radii, E_star, work_adhesion,
                                         order, bin_start, nbx, bin_size).max())
    if not np.isfinite(kmax) or kmax <= 0.0:
        return np.inf
    return safety * 2.0 * gamma_sub / kmax


def jkr_equilibrium_overlap(R, E_star, w):
    """Overlap at which the JKR force vanishes (the cohesive minimum).

    F = 0 when 4 E* a^3/(3R) = sqrt(8 pi w E* a^3), i.e.
    a^(3/2) = 3R sqrt(8 pi w E*) / (4 E*), from which delta follows.
    """
    a = ((3.0 * R * np.sqrt(8.0 * np.pi * w * E_star)) / (4.0 * E_star)) ** (2.0 / 3.0)
    c = np.sqrt(2.0 * np.pi * w / E_star)
    return float(a * a / R - c * np.sqrt(a))
