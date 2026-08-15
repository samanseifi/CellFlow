"""Does a SQUARE of cells relax into a disc under adhesion + repulsion alone?

The cleanest possible surface-tension test, and a better one than the 4:1
rectangle in giverso_surface_tension.py. A rectangle's aspect ratio can fall for
reasons that have nothing to do with capillarity (the pack simply spreading), and
aspect ratio cannot tell a square from a disc at all -- both are 1.0. A square
has what we actually want to probe: FLAT EDGES (zero curvature) and SHARP CORNERS
(high curvature). A surface tension acts on curvature, so it must eat the corners
and leave the edges, converging on a circle. Anything that merely relaxes the
pack uniformly will not.

No biology whatsoever: uptake, growth, division, death, chemotaxis and the random
walk are all off. Only adhesion, steric repulsion, and (optionally) the overlap
projection act.

Two observables, both robust for a convex shape:

  * circularity  C = 4 pi A / P^2  from the convex hull of the cell centres.
    Square = pi/4 = 0.785, disc = 1.000. Hull-based, so it is not vulnerable to
    the angular-binning error that made 4 pi A / P^2 useless for the rectangle.
  * a4, the four-fold Fourier amplitude of the boundary. A square's corners ARE
    its k=4 mode; a disc has none. a4 -> 0 is the signature of rounding, and it
    is independent of any perimeter estimate.

Three controls, because "it did not round" has more than one explanation:

  1. OVERLAP PROJECTION on/off. Issue #31 records that at relaxation 1.0 the
     projection overrides the force balance and leaves no surface tension at any
     adhesion strength. If the square only rounds with the projection off, the
     projection is the culprit rather than the force law.
  2. MOTILITY on/off. A dry adhesive granular pack has a YIELD STRESS: it can
     possess a surface tension and still not round, because nothing lets cells
     rearrange past each other. Adding a small random walk supplies that. If the
     square rounds only with motility, we have capillarity plus jamming -- a very
     different diagnosis from no capillarity at all.
  3. ADHESION SWEEP. If no value of adhesion_strength rounds the square, the
     force law cannot produce a surface tension, full stop.

Run:  python experiments/giverso_square_relax.py [steps]
"""
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.cell import Cell                                      # noqa: E402
from cellflow.analysis.front import (front_radii, main_cluster_mask,  # noqa: E402
                                     front_modes)
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Smaller than the dispersion regimes: a half-width of 80 units is 33 cell radii,
# so the corners are still resolved by ~30 cells while the run stays quick.
HALF = 80.0


def square_regime(**over):
    """PROPORTIONAL mechanics with every active process switched off."""
    r = dict(gd.PROPORTIONAL, name='square_relax', seed_eps=0.0,
             L=400.0, G=200, R0=HALF,
             consumption=0.0, basal=0.0, chi_nutrient=0.0,
             walk_speed=0.0, max_force=0.0)
    r.update(over)
    return r


def build_square(regime, rng, half=HALF):
    """Hex-packed square block of fixed-size, non-growing cells.

    ``spacing_factor`` matters more than it looks. The study's usual 1.9 puts
    neighbours at 1.9a against a touching distance of 2a, i.e. every pair starts
    5% overlapped. With the exponential repulsion law that is not a small
    detail: measured on this block the mean repulsive force is 56 against a mean
    adhesive force of 0.11, a factor of 500. The square would then be relaxing a
    compressed repulsive pack, and any shape change would be that pressure
    escaping rather than a surface tension acting. A factor of 2.0 starts every
    pair exactly at contact instead.
    """
    a = regime['cell_r']
    spacing = regime.get('spacing_factor', 1.9) * a
    Cell.next_id = 0
    cells = []
    c = regime['L'] / 2
    n = int(2.4 * half / spacing) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-n, n + 1):
            x = i * spacing + xoff + rng.normal(0, 0.10 * a)
            yy = y + rng.normal(0, 0.10 * a)
            if abs(x) <= half and abs(yy) <= half:
                cell = Cell(np.array([c + x, c + yy]), area_conserving=True)
                cell.max_radius = float(regime['max_radius'])
                cell.min_radius = 0.5 * cell.max_radius
                # radius must be consistent with stored nutrient (area-conserving)
                cell.nutrient_accumulated = 100.0 * (a / cell.max_radius) ** 2
                cell.radius = a
                cell.consumption_rate = 0.0
                cell.basal_metabolism_rate = 0.0
                cells.append(cell)
    return cells


def circularity(pos, rad):
    """4 pi A / P^2 from the convex hull. 0.785 for a square, 1.0 for a disc."""
    keep = main_cluster_mask(pos, rad)
    p = pos[keep]
    if len(p) < 4:
        return np.nan
    h = ConvexHull(p)
    # scipy: .volume is the enclosed AREA in 2D, .area is the PERIMETER
    return float(4.0 * np.pi * h.volume / (h.area ** 2))


def mode4(pos, rad, center):
    """Four-fold boundary amplitude -- the corners of the square."""
    keep = main_cluster_mask(pos, rad)
    r = front_radii(pos[keep], center, 360)
    return float(front_modes(r)[4])


def run(label, regime, steps, sample=None):
    sample = sample or max(1, steps // 5)
    sim = CellSimulation(gd.make_config(regime, 1), config_name='square_relax')
    sim.cells = build_square(regime, np.random.default_rng(1))
    sim.nutrient_field[:] = regime['bc_value']       # uniform; uptake is zero
    ctr = np.array([regime['L'] / 2, regime['L'] / 2])

    def obs():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        return circularity(pos, rad), mode4(pos, rad, ctr)

    c0, m0 = obs()
    snap0 = [(c.position.copy(), c.radius) for c in sim.cells]
    traj = [(0, c0, m0)]
    for s in range(steps):
        sim._simulation_step()
        if (s + 1) % sample == 0:
            cc, mm = obs()
            traj.append((s + 1, cc, mm))
    c1, m1 = traj[-1][1], traj[-1][2]
    snap1 = [(c.position.copy(), c.radius) for c in sim.cells]
    # Progress is measured from where this block STARTED, not from the ideal
    # square. A hex-packed jittered block already sits near 0.80 rather than
    # pi/4 = 0.785, because the packing itself blunts the corners a little.
    # Scoring against 0.785 would credit that as rounding the run did not do.
    prog = (c1 - c0) / max(1.0 - c0, 1e-9)
    print(f"  {label:<34} circularity {c0:.4f} -> {c1:.4f}  "
          f"({prog*100:+5.1f}% of the remaining way to a disc)   "
          f"a4 {m0:.4f} -> {m1:.4f} ({m1/max(m0,1e-9):.2f}x)  n={len(sim.cells)}",
          flush=True)
    return dict(label=label, traj=traj, c0=c0, c1=c1, m0=m0, m1=m1,
                prog=prog, snap0=snap0, snap1=snap1, L=regime['L'])


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"Square -> disc under adhesion + repulsion only. {steps} steps.")
    print("circularity: 0.785 = square, 1.000 = disc.  a4 = the corners.\n")

    results = []
    print("PRIMARY -- exactly as posed: no biology, default mechanics")
    results.append(run("square, default mechanics", square_regime(), steps))

    print("\nCONTROL 1 -- the overlap projection (issue #31)")
    for ov in (0, 1, 2):
        results.append(run(f"overlap_iterations = {ov}",
                           square_regime(overlap_iterations=ov), steps))

    # NOTE the propulsive forces here (1-5) are small against the ~56 mean
    # repulsive force in the pack, so this is a weak probe of jamming. It is
    # kept because Control 3 makes the jamming question secondary: with no
    # cohesive driving force there is nothing for jamming to be resisting.
    print("\nCONTROL 2 -- motility, so cells CAN rearrange (jamming test)")
    for w, f in ((0.02, 1.0), (0.05, 5.0)):
        results.append(run(f"random walk {w}, force {f}",
                           square_regime(walk_speed=w, max_force=f,
                                         propulsion_response='saturated'),
                           steps))

    print("\nCONTROL 3 -- adhesion sweep (default is 0.5)")
    for ad in (0.0, 10.0, 50.0):
        results.append(run(f"adhesion_strength = {ad}",
                           square_regime(adhesion=ad), steps))

    print("\nCONTROL 4 -- start UNCOMPRESSED (pairs exactly at contact)")
    print("  the 1.9 packing starts every pair 5% overlapped, giving a mean")
    print("  repulsive force of 56 against a mean adhesive force of 0.11")
    for ad in (0.5, 50.0):
        results.append(run(f"spacing 2.0, adhesion = {ad}",
                           square_regime(spacing_factor=2.0, adhesion=ad), steps))

    print("\n" + "=" * 78)
    best = max(results, key=lambda r: r['prog'])
    print(f"Best rounding: {best['label']} at {best['prog']*100:+.1f}% "
          f"of the way to a disc.")
    print("A genuine surface tension would take ANY of these to ~100%.")
    print("=" * 78)

    n = len(results)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.3 * ((n + 1) // 2), 7.2))
    for ax, r in zip(np.atleast_1d(axes).ravel(), results):
        for p, rr in r['snap1']:
            ax.add_patch(Circle(p, rr, color='#3b7dd8', lw=0))
        m = HALF * 1.7
        ax.set_xlim(r['L'] / 2 - m, r['L'] / 2 + m)
        ax.set_ylim(r['L'] / 2 - m, r['L'] / 2 + m)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r['label']}\nC {r['c0']:.3f}$\\to${r['c1']:.3f}  "
                     f"({r['prog']*100:+.0f}%)", fontsize=8)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis('off')
    fig.suptitle("Square relaxing under adhesion + repulsion only (no biology).\n"
                 "Circularity 0.785 = square, 1.000 = disc.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(HERE, 'giverso_square_relax.png')
    fig.savefig(out, dpi=115)
    print(f"\nSaved -> {out}")
    return results


if __name__ == '__main__' and not ({'trapped','screening','friction'} & set(sys.argv)):
    main()


def trapped_test(steps=3000):
    """Is the square TRAPPED rather than driven by nothing?

    The energy accounting says the disc is genuinely the lower state -- it has
    +13 more bonds out of 4646 -- so a driving force exists. But two things make
    it unreachable:

      * it is tiny. At the default adhesion the whole square->disc gain is ~10
        energy units against a barrier of ~17.6 for a SINGLE cell rearrangement,
        and it sits far below the +-740 scatter that packing disorder alone
        contributes.
      * the dynamics is deterministic and overdamped. x' = F/gamma slides to the
        NEAREST local minimum and stops; it cannot cross a barrier of any height
        because there is no noise. A jittered square pack is already a local
        minimum -- every cell sits in a well formed by its neighbours.

    So the earlier motility control (forces 1-5) was far too weak to test this:
    the relevant scale is the ~17.6 barrier and the ~56 repulsive force, not 1.
    This sweep drives the noise up to and past that scale. If the square rounds
    at strong noise, the correct diagnosis is "surface tension present but
    kinetically frozen", NOT "no surface tension" -- and the fix then needs a
    rearrangement mechanism, not only a deeper adhesive well.
    """
    print("\nTRAPPED-OR-ABSENT: noise at the scale of the rearrangement barrier")
    print("  (barrier ~17.6, mean repulsive force ~56; earlier control used 1-5)")
    out = []
    for ad in (0.5, 50.0):
        for f in (20.0, 60.0, 150.0):
            out.append(run(f"adh {ad}, walk force {f}",
                           square_regime(adhesion=ad, walk_speed=0.05,
                                         max_force=f,
                                         propulsion_response='saturated'),
                           steps))
    return out


if __name__ == '__main__' and 'trapped' in sys.argv:
    trapped_test(int(sys.argv[1]) if sys.argv[1].isdigit() else 3000)


def screening_test(steps=3000):
    """The cheapest decisive test: is the failure the SCREENING LENGTH?

    Cell velocities come from the Brinkman solve, whose transfer function
    u_hat(k) = f_hat(k)/(mu k^2 + alpha), alpha = mu/delta^2, suppresses
    cell-scale forces by (mu k^2 + alpha)/alpha. At the shipped delta = 14
    (6.5 cell radii) that is 416x at the cell scale -- and rounding a corner IS
    a cell-scale rearrangement.

    But delta = 14 is not obviously the right value. Physically the screening in
    a dense pack is set by the PORE scale, i.e. delta ~ one cell radius, not six.
    At delta = a = 2.16 the cell-scale suppression falls from 416x to ~11x.

    So this costs one config value and separates two very different diagnoses:
      * square rounds at small delta  -> the formulation is fine, the SCREENING
        LENGTH was unphysical, and much of this study ran with cell-scale motion
        suppressed by two orders of magnitude for no good reason.
      * square still does not round   -> the parameter is exonerated and the
        problem is the formulation itself (v_cell = u_fluid forbids neighbour
        exchange at any delta), which needs a local friction law instead.
    Run with both weak and strong adhesion, since mobility and cohesion are the
    two candidate binding constraints and we need to know which is binding.
    """
    print("\nSCREENING-LENGTH TEST: is cell-scale motion just over-damped?")
    print("  suppression at the cell scale = (mu k^2 + alpha)/alpha")
    mu = 500.0
    k_cell = 2 * np.pi / (2 * 2.16)
    for d in (2.16, 7.0, 14.0):
        al = mu / d ** 2
        print(f"    delta={d:5.2f} ({d/2.16:.1f} cell radii): "
              f"{(mu*k_cell**2+al)/al:6.1f}x suppression")
    print()
    out = []
    for d in (2.16, 7.0, 14.0):
        for ad in (0.5, 50.0):
            out.append(run(f"delta={d}, adhesion={ad}",
                           square_regime(screening=d, adhesion=ad), steps))
    print("\n  (a square that rounds shows circularity -> 1.0 and a4 -> 0)")
    return out


if __name__ == '__main__' and 'screening' in sys.argv:
    screening_test(3000)


def friction_test(steps=3000):
    """ACCEPTANCE GATE for issue #32: does the local friction law round a square?

    Under the fluid law the answer is no, and adhesion strength makes no
    difference, because v_cell = u_fluid(x_cell) advects every cell by one
    smooth field and neighbours can never exchange places. The friction law
    solves gamma_sub v_i + sum_j gamma_cc w_ij (v_i - v_j) = F_i instead, which
    is local and does permit relative motion.

    The gate has two halves, and both must pass:
      1. the square must actually round (circularity -> 1, a4 -> 0);
      2. it must round FASTER with stronger adhesion, since it is the adhesive
         well that supplies the driving force.
    Half 2 matters as much as half 1: rounding that does not respond to adhesion
    is the pack relaxing, not a surface tension.
    """
    print("\nACCEPTANCE GATE (#32): local friction velocity law")
    print("  gamma_sub = substrate drag; gamma_cc = cell-cell friction\n")
    out = []
    for ad in (0.0, 0.5, 5.0, 50.0):
        out.append(run(f"friction, adhesion={ad}",
                       square_regime(velocity_model='friction',
                                     friction_substrate=1.0,
                                     friction_cell_cell=0.5,
                                     adhesion=ad), steps))
    print()
    for gcc in (0.0, 5.0):
        out.append(run(f"friction gamma_cc={gcc}, adhesion=5.0",
                       square_regime(velocity_model='friction',
                                     friction_substrate=1.0,
                                     friction_cell_cell=gcc,
                                     adhesion=5.0), steps))
    print("\n  PASS requires: circularity rising toward 1.0, AND more adhesion")
    print("  giving more rounding.")
    return out


if __name__ == '__main__' and 'friction' in sys.argv:
    friction_test(int(sys.argv[1]) if sys.argv[1].isdigit() else 3000)
