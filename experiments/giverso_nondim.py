"""Non-dimensional setup for the Giverso branching replication.

The continuum model is written in units of the NUTRIENT DIFFUSION LENGTH
  l = sqrt(D_n / k)
(its dimensionless nutrient obeys  n_dot = lap(n) - n  inside the colony, i.e.
length scaled by l). Branching is "strongly diffusion-limited" when the colony
radius R >> l. So to map our discrete parameters onto the paper's regime we
must (1) measure l for our (nutrient_D, consumption, cell density), and (2) size
the run so R/l is large.

This module:
  - measure_diffusion_length(): pack a static disk of cells, hold a Dirichlet
    nutrient reservoir, iterate diffuse+absorb to steady state, and fit the
    radial nutrient profile n(r) ~ exp(-(R-r)/l) just inside the rim -> l.
  - front_power_spectrum(): extract the colony front R(theta) and return its
    angular power spectrum (the discrete analog of the paper's dispersion
    curve); the dominant mode = number of fingers.

Run:  python experiments/giverso_nondim.py        # measure l for a few D values
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.cell import Cell                                   # noqa: E402
from cellflow.kernels.diffusion import diffuse_field_numba       # noqa: E402
from cellflow.kernels.fields import absorb_nutrient_numba        # noqa: E402


def packed_disk(Rc, cell_r, L, spacing=None):
    """A hex-packed static disk of cells of radius cell_r and radius Rc at center."""
    if spacing is None:
        spacing = 1.9 * cell_r          # nearly touching
    Cell.next_id = 0
    cells = []
    c = L / 2
    ny = int(2 * Rc / (spacing * np.sqrt(3) / 2)) + 2
    for j in range(-ny, ny + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        nx = int(2 * Rc / spacing) + 2
        for i in range(-nx, nx + 1):
            x = i * spacing + xoff
            if x * x + y * y <= Rc * Rc:
                cell = Cell(np.array([c + x, c + y]))
                cell.radius = cell_r
                cells.append(cell)
    return cells


def measure_diffusion_length(nutrient_D, consumption=0.2, cell_r=2.4,
                             Rc=40.0, L=200.0, G=200, dt=0.05, bc=40.0, iters=1500):
    """Steady-state radial nutrient decay length inside a packed colony."""
    dx = L / G
    cells = packed_disk(Rc, cell_r, L)
    for cobj in cells:
        cobj.consumption_rate = consumption
    field = np.full((G, G), bc)
    for _ in range(iters):
        read = np.copy(field)
        for cobj in cells:
            absorb_nutrient_numba(cobj.position, cobj.radius, field, read,
                                  dt, cobj.consumption_rate, dx)
        field = diffuse_field_numba(field, nutrient_D, dt, dx, bc)

    # radial average about the centre
    c = L / 2
    yy, xx = np.mgrid[0:G, 0:G] * dx
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    rbins = np.arange(0, Rc, dx)
    prof = np.array([field[(r >= rb) & (r < rb + dx)].mean() for rb in rbins])
    depth = Rc - rbins                          # depth below the rim
    # The interior plateaus at n_inf (first-order consumption balances the
    # residual diffusive flux). Fit n - n_inf ~ A exp(-depth/l) from the rim in.
    n_inf = prof[:3].mean()
    excess = prof - n_inf
    mask = excess > 0.05 * (prof.max() - n_inf)
    if mask.sum() < 3:
        return np.nan, rbins, prof
    coeffs = np.polyfit(depth[mask], np.log(excess[mask]), 1)
    ell = -1.0 / coeffs[0] if coeffs[0] < 0 else np.nan
    return ell, rbins, prof


def front_power_spectrum(positions, center, n_bins=180):
    """Colony front R(theta) and its angular power spectrum.
    Returns (modes, power, R_theta). Dominant nonzero mode ~ number of fingers."""
    d = positions - center
    theta = np.arctan2(d[:, 1], d[:, 0])
    rad = np.hypot(d[:, 0], d[:, 1])
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    Rth = np.full(n_bins, np.nan)
    idx = np.digitize(theta, edges) - 1
    for b in range(n_bins):
        rr = rad[idx == b]
        if rr.size:
            Rth[b] = rr.max()                   # outermost cell in this wedge
    # fill gaps
    good = ~np.isnan(Rth)
    Rth = np.interp(np.arange(n_bins), np.where(good)[0], Rth[good], period=n_bins)
    f = np.fft.rfft(Rth - Rth.mean())
    power = np.abs(f) ** 2
    modes = np.arange(power.size)
    return modes, power, Rth


def main():
    print("Nutrient diffusion length l = sqrt(D/k_eff), measured in a packed colony")
    print(f"{'nutrient_D':>11} {'l (units)':>10} {'R=40 -> R/l':>14}")
    for D in (0.01, 0.03, 0.1, 0.3):
        ell, _, _ = measure_diffusion_length(D)
        print(f"{D:>11.3f} {ell:>10.2f} {40.0/ell:>14.1f}")
    print("\nFor strongly diffusion-limited branching the paper needs R/l >> 1.")
    print("Pick D (and colony size) so the final R/l is ~10-20.")


if __name__ == '__main__':
    main()
