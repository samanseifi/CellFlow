"""Fast regime check: what (nutrient_D, consumption) gives a THIN active rim?

No growth, no division, no fluid -- just a frozen hex disk of cells, the nutrient
field diffused to quasi-steady with first-order uptake, and a readout of how deep
the nutrient penetrates.  The branching mechanism needs an active rim thin
compared to the colony (R/l large): if ~100% of the colony stays above the
quiescence threshold, the whole bulk grows and the front just rounds out (what we
keep seeing).  We want the active fraction down to ~10-20%.

Run:  python experiments/giverso_lcheck.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.kernels.diffusion import diffuse_field_numba          # noqa: E402
from cellflow.kernels.fields import absorb_nutrient_numba           # noqa: E402

L = 400.0
G = 300                    # dx ~ 1.33
DX = L / G
DT = 0.05
BC = 100.0
R = 80.0                   # frozen colony radius
CELL_R = 2.4
THRESH = 25.0              # quiescence threshold
RELAX = 1200               # diffusion steps to quasi-steady (need many at high D)


def hex_disk(R, cell_r=CELL_R):
    spacing = 1.9 * cell_r
    c = L / 2
    pts = []
    n = int(2 * R / (spacing * np.sqrt(3) / 2)) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-int(2 * R / spacing) - 2, int(2 * R / spacing) + 2):
            x = i * spacing + xoff
            if np.hypot(x, y) <= R:
                pts.append((c + x, c + y))
    return np.array(pts)


def equilibrate(positions, D, cons):
    field = np.full((G, G), BC)
    for _ in range(RELAX):
        field = diffuse_field_numba(field, D, DT, DX, BC)
        read = field.copy()
        for p in positions:
            absorb_nutrient_numba(np.array(p), CELL_R, field, read, DT, cons, DX)
    return field


def report(positions, field, D, cons):
    c = L / 2
    # field sampled at each cell centre
    vals = np.array([field[int(p[1] / DX), int(p[0] / DX)] for p in positions])
    dist = np.array([np.hypot(p[0] - c, p[1] - c) for p in positions])
    active = vals > THRESH
    frac = active.mean()
    # rim thickness: R minus innermost active radius
    rim = R - dist[active].min() if active.any() else 0.0
    n_center = field[int(c / DX), int(c / DX)]
    # penetration l: fit field(dist from rim) ~ BC*(1-exp(-(R-dist)/l)) inside
    inside = dist < R - 1
    order = np.argsort(dist[inside])
    print(f"  D={D:<6} cons={cons:<5}: active {100*frac:5.1f}%  "
          f"rim {rim:5.1f}u ({rim/CELL_R:4.1f} cell-R)  "
          f"n(center)={n_center:6.2f}  R/rim~{R/rim if rim>0 else float('inf'):.0f}")
    return frac, rim


def main():
    pos = hex_disk(R)
    print(f"frozen disk: {len(pos)} cells, R={R} ({R/CELL_R:.0f} cell radii), "
          f"threshold={THRESH}, BC={BC}")
    print("sweeping (nutrient_D, consumption) -> active fraction & rim thickness:")
    print(f"(dt={DT}, dx={DX:.2f}; explicit-diffusion stable for D < {DX**2/(4*DT):.1f})")
    for D in (1.0, 3.0, 6.0):
        for cons in (0.2, 0.6, 1.5, 4.0):
            field = equilibrate(pos, D, cons)
            report(pos, field, D, cons)
    print("\nTarget: active ~10-20%, rim a few cell radii, R/rim large (~10-20).")


if __name__ == '__main__':
    main()
