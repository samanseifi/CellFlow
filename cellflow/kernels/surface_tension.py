"""Explicit interfacial surface tension: Young-Laplace on the colony boundary.

Issue #36. A surface tension does not emerge from this model's pair potential --
the square test shows the driving force is 13 bonds in 4646 and the pack jams
before it can find them (#40) -- so this imposes the interfacial mechanics
directly, where it physically lives.

Approach
--------
The obvious route is a continuum surface force on the grid (Brackbill CSF):
smooth an occupancy field, take ``kappa = -div(grad phi/|grad phi|)``, apply
``sigma kappa n |grad phi|``. That was implemented first and rejected on
measurement: curvature is a second derivative, the cell-scale roughness of a
packed boundary dominates it, and ``kappa * R`` for a disc came out anywhere
between 0.07 and 1.33 depending on colony size and smoothing width. No setting
was accurate across sizes.

Instead the boundary is parameterised as ``R(theta)`` about the colony centroid
and band-limited by truncating its Fourier series. Curvature then follows in
closed form,

    kappa = (R^2 + 2 R'^2 - R R'') / (R^2 + R'^2)^(3/2)

which is exact for the truncated shape rather than a finite-difference estimate
of a noisy one. Truncation is the physically right regularisation here: the
dispersion analysis cares about modes k <~ 20, and modes at the cell scale are
not interface shape at all. It also makes the smoothing an explicit, reportable
parameter (``k_max``) rather than a grid artifact.

The force ``sigma * kappa`` is applied inward along the local normal to the
boundary cells only, reproducing ``p = p0 - sigma_b C`` -- Giverso et al.'s
closure.

Limitation, stated plainly: R(theta) requires a star-shaped colony about its
centroid. That holds for the fronts this study measures (lobed discs up to
deep-necked fingers) but would fail for an overhanging or fragmented shape, and
the caller is expected to check.

Scope note: with this active a front instability is no longer *emergent from
cell-scale rules* -- the mechanism is imposed, exactly as it is in the continuum
model being compared against. See docs/giverso_replication.md.
"""
import numpy as np


def boundary_profile(positions, center, n_bins=360):
    """Outermost cell radius per angular bin: R(theta), plus bin occupancy.

    Empty bins are filled by circular interpolation from their neighbours so the
    Fourier transform below sees a complete profile.
    """
    d = positions - center
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * np.pi)
    idx = np.minimum((th / (2.0 * np.pi) * n_bins).astype(int), n_bins - 1)
    R = np.zeros(n_bins)
    np.maximum.at(R, idx, r)
    filled = R > 0
    if not filled.any():
        return R, filled
    if not filled.all():
        angles = (np.arange(n_bins) + 0.5) * 2.0 * np.pi / n_bins
        R[~filled] = np.interp(angles[~filled], angles[filled], R[filled],
                               period=2.0 * np.pi)
    return R, filled


def truncate_modes(R, k_max):
    """Band-limit R(theta) to angular modes <= k_max."""
    F = np.fft.rfft(R)
    F[k_max + 1:] = 0.0
    return np.fft.irfft(F, n=len(R))


def curvature_from_profile(R, k_max=20):
    """Signed curvature of the band-limited boundary; positive where convex.

    Derivatives are taken spectrally on the truncated series, so they are exact
    for that shape rather than differenced.
    """
    n = len(R)
    Rs = truncate_modes(R, k_max)
    k = np.fft.rfftfreq(n, d=1.0 / n)
    F = np.fft.rfft(Rs)
    dR = np.fft.irfft(1j * k * F, n=n)
    d2R = np.fft.irfft(-(k ** 2) * F, n=n)
    num = Rs ** 2 + 2.0 * dR ** 2 - Rs * d2R
    den = (Rs ** 2 + dR ** 2) ** 1.5
    return num / np.maximum(den, 1e-30), Rs, dR


def surface_tension_forces(positions, radii, center, sigma, k_max=20,
                           n_bins=360, boundary_frac=0.9):
    """Per-cell Young-Laplace force, applied to boundary cells only.

    Parameters
    ----------
    boundary_frac : float
        A cell counts as interface if its radius exceeds this fraction of the
        local R(theta). Interior cells get zero force, which is what makes this
        a SURFACE term rather than a body force.

    Returns
    -------
    forces : (N, 2) array
    info : dict with the mean curvature and the boundary-cell count.
    """
    n = len(positions)
    forces = np.zeros((n, 2))
    if n == 0:
        return forces, {'mean_curvature': 0.0, 'n_boundary': 0}

    R, _ = boundary_profile(positions, center, n_bins)
    kappa, Rs, dR = curvature_from_profile(R, k_max)

    d = positions - center
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * np.pi)
    idx = np.minimum((th / (2.0 * np.pi) * n_bins).astype(int), n_bins - 1)

    on_boundary = r >= boundary_frac * Rs[idx]
    if not on_boundary.any():
        return forces, {'mean_curvature': float(kappa.mean()), 'n_boundary': 0}

    # Outward normal of the curve r = R(theta): proportional to
    # (R e_r - R' e_theta), normalised.
    er = np.stack([np.cos(th), np.sin(th)], axis=1)
    et = np.stack([-np.sin(th), np.cos(th)], axis=1)
    nvec = Rs[idx, None] * er - dR[idx, None] * et
    nvec /= np.maximum(np.linalg.norm(nvec, axis=1, keepdims=True), 1e-30)

    # Young-Laplace: pressure excess sigma*kappa inside pushes the surface
    # INWARD where the boundary is convex, which is what shrinks the perimeter.
    mag = sigma * kappa[idx]
    forces[on_boundary] = -(mag[on_boundary, None] * nvec[on_boundary])
    return forces, {'mean_curvature': float(np.mean(kappa)),
                    'n_boundary': int(on_boundary.sum())}


def laplace_pressure(sigma, radius):
    """Young-Laplace pressure jump across a 2D circular interface: dp = sigma/R."""
    return sigma / radius
