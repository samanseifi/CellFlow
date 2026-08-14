"""Quantitative diagnostics for a colony front.

Measuring whether a growing colony front is *linearly unstable* is easy to get
wrong. The failure mode that produced a retracted result in this repo (see
``docs/giverso_replication.md``) is worth stating up front, because these
functions exist to prevent it:

    The front radius is naturally defined as the OUTERMOST cell in each angular
    bin. If cells detach from the colony -- which they do whenever adhesion is
    weak and the rim is starving -- a handful of strays per bin inflate that
    radius. The front then looks rough, and a seeded mode looks amplified, while
    the actual connected colony is smooth and the mode is decaying.

So: **always filter to the main connected cluster before extracting a front.**
:func:`analyze_front` does this by construction; the lower-level pieces are
exposed for tests.

The second failure mode is measuring an "instability" on a front that is not
advancing. Mullins-Sekerka is an instability of a *growing* front; a colony that
is eroding can roughen for entirely different reasons (here, starvation death
pitting the rim). Callers should check ``mean_radius`` is increasing --
:func:`fit_growth_rate` reports the fit quality but cannot know this for you.
"""
import numpy as np
from numba import njit

from ..kernels.neighbors import build_cell_list_numba


@njit(cache=True)
def _find(parent, x):
    """Union-find root with path halving (iterative -- numba has no recursion)."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _connected_components(positions, radii, link_factor,
                          order, bin_start, nbx, bin_size):
    """Label connected components; two cells are linked when

        |x_i - x_j| < link_factor * (r_i + r_j)

    i.e. ``link_factor = 1.0`` means "touching". Uses the cell list so the pair
    scan is O(N) rather than O(N^2); union-find is sequential (not thread safe).
    """
    n = positions.shape[0]
    parent = np.arange(n)

    for i in range(n):
        bx = int(positions[i, 0] / bin_size)
        by = int(positions[i, 1] / bin_size)
        if bx < 0:
            bx = 0
        elif bx >= nbx:
            bx = nbx - 1
        if by < 0:
            by = 0
        elif by >= nbx:
            by = nbx - 1
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
                    if j <= i:
                        continue
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    reach = link_factor * (radii[i] + radii[j])
                    if dx * dx + dy * dy < reach * reach:
                        ri = _find(parent, i)
                        rj = _find(parent, j)
                        if ri != rj:
                            if ri < rj:
                                parent[rj] = ri
                            else:
                                parent[ri] = rj

    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        labels[i] = _find(parent, i)
    return labels


def main_cluster_mask(positions, radii, link_factor=1.3):
    """Boolean mask selecting cells in the LARGEST connected cluster.

    Parameters
    ----------
    positions : (N, 2) array
    radii : (N,) array
    link_factor : float
        Two cells are connected when their centre distance is less than
        ``link_factor * (r_i + r_j)``. ``1.0`` is exactly touching; the default
        ``1.3`` tolerates the small gaps of a normally-packed colony while still
        rejecting cells that have genuinely drifted off the rim.

    Returns
    -------
    (N,) bool array. All-``True`` for an empty or single-cell input.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    n = positions.shape[0]
    if n <= 1:
        return np.ones(n, dtype=bool)

    # Bin size must cover the longest possible link so no connected pair is missed.
    bin_size = 2.0 * link_factor * float(radii.max())
    # build_cell_list_numba clamps negative coordinates into bin 0, which is
    # correct but degenerate; shift to the non-negative quadrant so the binning
    # is actually spatial. Distances are translation invariant.
    shifted = np.ascontiguousarray(positions - positions.min(axis=0))
    extent = float(shifted.max()) + bin_size
    order, bin_start, nbx = build_cell_list_numba(shifted, extent, bin_size)
    labels = _connected_components(shifted, radii, float(link_factor),
                                   order, bin_start, nbx, bin_size)
    uniq, counts = np.unique(labels, return_counts=True)
    return labels == uniq[np.argmax(counts)]


def front_radii(positions, center, n_bins=360):
    """Outer front radius R(theta) sampled on ``n_bins`` equal angular bins.

    Takes the outermost cell in each bin. **Filter to the main connected cluster
    first** (see module docstring) or detached cells will dominate the result.
    Empty bins are filled by periodic interpolation from their neighbours.

    Returns
    -------
    (n_bins,) array of radii, ordered by angle from -pi to pi.
    """
    positions = np.asarray(positions, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if positions.shape[0] == 0:
        raise ValueError("front_radii needs at least one cell")

    d = positions - center
    theta = np.arctan2(d[:, 1], d[:, 0])
    rad = np.hypot(d[:, 0], d[:, 1])

    idx = np.clip(((theta + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int64),
                  0, n_bins - 1)
    R = np.full(n_bins, -np.inf)
    np.maximum.at(R, idx, rad)

    good = np.isfinite(R)
    if not good.any():
        raise ValueError("no cells landed in any angular bin")
    if not good.all():
        R = np.interp(np.arange(n_bins), np.flatnonzero(good), R[good],
                      period=n_bins)
    return R


def front_modes(R):
    """Dimensionless angular-mode amplitudes of a front R(theta).

    Returns ``a_k = 2 |FFT_k(R - <R>)| / (n <R>)`` for k = 0, 1, 2, ..., so
    ``a_k`` is the amplitude of mode k as a *fraction of the mean radius* -- the
    quantity whose growth rate is the linear-stability eigenvalue. ``a_0`` is 0
    by construction (the mean is removed).
    """
    R = np.asarray(R, dtype=np.float64)
    mean = R.mean()
    if mean <= 0.0:
        raise ValueError("front radii must have a positive mean")
    return 2.0 * np.abs(np.fft.rfft(R - mean)) / (len(R) * mean)


def roughness(R):
    """Dimensionless front roughness, std(R) / mean(R)."""
    R = np.asarray(R, dtype=np.float64)
    return float(np.std(R) / np.mean(R))


def analyze_front(positions, radii, center, n_bins=360, link_factor=1.3):
    """Full front diagnostic on the main connected cluster.

    Returns a dict with:
        ``n_cells``      total cells in
        ``n_detached``   cells excluded as not part of the main cluster
        ``mean_radius``  <R> of the connected front (check this is INCREASING)
        ``roughness``    std(R)/mean(R) of the connected front
        ``modes``        dimensionless mode amplitudes (see :func:`front_modes`)
        ``radii_profile`` the R(theta) array itself
    """
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    keep = main_cluster_mask(positions, radii, link_factor)
    R = front_radii(positions[keep], center, n_bins)
    return {
        'n_cells': int(positions.shape[0]),
        'n_detached': int((~keep).sum()),
        'mean_radius': float(R.mean()),
        'roughness': roughness(R),
        'modes': front_modes(R),
        'radii_profile': R,
    }


def fit_growth_rate(times, amplitudes):
    """Fit an exponential growth rate lambda to a mode amplitude time series.

    A linear instability grows as ``a(t) = a_0 exp(lambda t)``, so lambda is the
    slope of ``log a`` against ``t``. Fitting the *log* (rather than taking an
    end-to-start ratio) is what distinguishes genuine exponential growth from a
    transient excursion: a non-exponential series shows up as a poor ``r_squared``
    even when the endpoints happen to be higher than the start.

    Non-positive amplitudes are dropped (log undefined). Requires >= 3 usable
    points; returns ``lambda = nan`` otherwise.

    Returns
    -------
    dict with ``lambda_``, ``stderr``, ``r_squared``, ``n_points``.
    """
    t = np.asarray(times, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)
    if t.shape != a.shape:
        raise ValueError("times and amplitudes must have the same shape")

    ok = np.isfinite(a) & (a > 0.0) & np.isfinite(t)
    t, a = t[ok], a[ok]
    n = t.size
    if n < 3:
        return {'lambda_': np.nan, 'stderr': np.nan, 'r_squared': np.nan,
                'n_points': int(n)}

    y = np.log(a)
    tbar, ybar = t.mean(), y.mean()
    stt = np.sum((t - tbar) ** 2)
    if stt <= 0.0:
        return {'lambda_': np.nan, 'stderr': np.nan, 'r_squared': np.nan,
                'n_points': int(n)}

    slope = np.sum((t - tbar) * (y - ybar)) / stt
    resid = y - (ybar + slope * (t - tbar))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - ybar) ** 2))
    stderr = np.sqrt(ss_res / (n - 2) / stt) if n > 2 else np.nan
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

    return {'lambda_': float(slope), 'stderr': float(stderr),
            'r_squared': float(r2), 'n_points': int(n)}
