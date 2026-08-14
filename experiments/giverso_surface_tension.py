"""Do we have the two physics Giverso relies on: a sharp interface and a
Young-Laplace surface tension? Measure both directly, with growth switched off.

Giverso's model closes the free-boundary problem with p = p0 - sigma_b C at a
SHARP interface. CellFlow has neither explicitly: its boundary is a few cell
layers thick and any surface tension is emergent from adhesion + steric
repulsion. Whether those emergent mechanics actually behave like a surface
tension is testable, and this is the test.

Switching growth off is what makes it clean. The paper's dispersion relation then
loses every term except the capillary one:

    lambda(k) = -(sigma / R*^3) k (k^2 - 1)

So the k-dependence of the DECAY rate is a fingerprint:

    lambda ~ -k(k^2-1)   => a genuine Young-Laplace surface tension,
                            and the prefactor gives sigma directly
    lambda ~ -k          => local rearrangement, NOT surface tension
                            (this is the shape we measured with growth ON)

Three measurements:
  A. Rectangle -> disc?  A drop with surface tension minimises perimeter at
     fixed area. Start from a rectangle with everything off and watch the
     circularity 4 pi A / P^2 relax toward 1.
  B. Mode-k decay.  Seed R(theta) = R0(1 + eps cos k theta) on a non-growing
     colony, fit lambda(k), and compare -k(k^2-1) against -k.
  C. Interface width.  How many cell diameters does the density take to fall
     from 90% to 10%? Sharp compared with the diffusion length, or not?

Run:  python experiments/giverso_surface_tension.py [steps]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.cell import Cell                                      # noqa: E402
from cellflow.analysis.front import (analyze_front, front_radii,    # noqa: E402
                                     main_cluster_mask, front_modes,
                                     fit_growth_rate)
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
L_DIFF = 8.9


def frozen_regime(**over):
    """PROPORTIONAL geometry with ALL activity off: no uptake, no growth, no
    division, no chemotaxis, no random walk. Only adhesion, repulsion and the
    overlap projection remain -- exactly the mechanics whose surface tension we
    are trying to measure."""
    r = dict(gd.PROPORTIONAL, name='surften', seed_eps=0.0,
             consumption=0.0, basal=0.0, chi_nutrient=0.0,
             walk_speed=0.0, max_force=0.0)
    r.update(over)
    return r


def build(regime, shape, rng, mode=0, eps=0.0, aspect=4.0):
    """Colony of fixed-size cells, either a disc/lobed disc or a rectangle of
    the same area."""
    R0, a = regime['R0'], regime['cell_r']
    spacing = 1.9 * a
    Cell.next_id = 0
    cells = []
    c = regime['L'] / 2
    area = np.pi * R0 ** 2
    half_h = np.sqrt(area / aspect) / 2.0
    half_w = aspect * half_h
    n = int(2.2 * max(R0, half_w) / spacing) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-n, n + 1):
            x = i * spacing + xoff + rng.normal(0, 0.12 * a)
            yy = y + rng.normal(0, 0.12 * a)
            if shape == 'rect':
                inside = abs(x) <= half_w and abs(yy) <= half_h
            else:
                rr, th = np.hypot(x, yy), np.arctan2(yy, x)
                inside = rr <= R0 * (1.0 + eps * np.cos(mode * th))
            if inside:
                cell = Cell(np.array([c + x, c + yy]), area_conserving=True)
                cell.max_radius = float(regime['max_radius'])
                cell.min_radius = 0.5 * cell.max_radius
                cell.nutrient_accumulated = 100.0 * (a / cell.max_radius) ** 2
                cell.radius = a
                cell.consumption_rate = 0.0
                cell.basal_metabolism_rate = 0.0
                cells.append(cell)
    return cells


def aspect_ratio(pos, rad):
    """Elongation from the second-moment tensor: 1 for a disc, 4 for a 4:1
    rectangle.

    An equal-ANGLE front sampling (front_radii) is a poor description of an
    elongated shape -- most bins land on the short ends, the polygon perimeter is
    badly overestimated, and 4 pi A / P^2 comes out at 0.09 for a rectangle whose
    true value is 0.50. The moment tensor has no such bias.
    """
    keep = main_cluster_mask(pos, rad)
    p = pos[keep]
    d = p - p.mean(axis=0)
    ev = np.linalg.eigvalsh(np.cov(d.T))
    return float(np.sqrt(max(ev) / max(min(ev), 1e-12))), p.mean(axis=0)


def make_sim(regime, cells):
    sim = CellSimulation(gd.make_config(regime, 1), config_name='surften')
    sim.cells = cells
    sim.nutrient_field[:] = regime['bc_value']     # uniform, unused (uptake=0)
    return sim


# ---------------------------------------------------------------------------
def test_rectangle(regime, steps):
    print("A. rectangle -> disc?  (everything off; only adhesion/repulsion act)")
    cells = build(regime, 'rect', np.random.default_rng(1), aspect=4.0)
    sim = make_sim(regime, cells)
    pos = np.array([c.position for c in sim.cells])
    rad = np.array([c.radius for c in sim.cells])
    c0, _ = aspect_ratio(pos, rad)
    snaps = [(0, [(c.position.copy(), c.radius) for c in sim.cells])]
    traj = [(0, c0)]
    for s in range(steps):
        sim._simulation_step()
        if (s + 1) % max(1, steps // 5) == 0:
            pos = np.array([c.position for c in sim.cells])
            rad = np.array([c.radius for c in sim.cells])
            cc, _ = aspect_ratio(pos, rad)
            traj.append((s + 1, cc))
            print(f"    step {s+1:5d}  aspect ratio {cc:.3f}", flush=True)
    snaps.append((steps, [(c.position.copy(), c.radius) for c in sim.cells]))
    print(f"  aspect ratio {c0:.3f} -> {traj[-1][1]:.3f}   "
          f"(1.0 = disc, 4.0 = the 4:1 rectangle we started from)")
    print("  => a surface tension would drive this toward 1.\n")
    return traj, snaps


def test_mode_decay(regime, steps, modes=(2, 3, 4, 6, 8, 10)):
    print("B. mode-k decay on a NON-growing colony")
    print("   Giverso with growth off:  lambda = -(sigma/R*^3) k(k^2-1)")
    out = {}
    for k in modes:
        cells = build(regime, 'disc', np.random.default_rng(1), mode=k, eps=0.12)
        sim = make_sim(regime, cells)
        ctr = np.array([regime['L'] / 2, regime['L'] / 2])
        ts, amps = [], []
        for s in range(steps + 1):
            if s % max(1, steps // 12) == 0:
                pos = np.array([c.position for c in sim.cells])
                rad = np.array([c.radius for c in sim.cells])
                keep = main_cluster_mask(pos, rad)
                a = front_modes(front_radii(pos[keep], ctr, 360))[k]
                ts.append(s * regime['dt']); amps.append(a)
            if s < steps:
                sim._simulation_step()
        fit = fit_growth_rate(ts, amps)
        out[k] = fit['lambda_']
        print(f"    k={k:2d}: lambda = {fit['lambda_']:+.5f}  "
              f"(r2={fit['r_squared']:.2f})  a: {amps[0]:.4f} -> {amps[-1]:.4f}",
              flush=True)
    return out


def test_interface_width(regime):
    print("\nC. interface width")
    cells = build(regime, 'disc', np.random.default_rng(1))
    pos = np.array([c.position for c in cells])
    ctr = np.array([regime['L'] / 2, regime['L'] / 2])
    r = np.hypot(*(pos - ctr).T)
    edges = np.arange(0, regime['R0'] * 1.3, regime['cell_r'] / 2)
    cnt, _ = np.histogram(r, bins=edges)
    ring = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    dens = cnt / ring
    dens /= np.median(dens[(edges[:-1] > 0.3 * regime['R0']) &
                           (edges[:-1] < 0.7 * regime['R0'])])
    ctrs = 0.5 * (edges[1:] + edges[:-1])
    # walk outward from the plateau: the density is noisy at r -> 0 (tiny rings),
    # so a global "first bin below 10%" picks up the centre and returns a
    # negative width.
    plateau = int(np.argmin(np.abs(ctrs - 0.5 * regime['R0'])))
    hi = lo = np.nan
    for i in range(plateau, len(ctrs)):
        if np.isnan(hi) and dens[i] < 0.9:
            hi = ctrs[i]
        if not np.isnan(hi) and dens[i] < 0.1:
            lo = ctrs[i]
            break
    w = lo - hi
    print(f"    density 90% -> 10% over {w:.1f} units "
          f"= {w/regime['cell_r']:.1f} cell radii = {w/L_DIFF:.2f} l")
    print(f"    (Giverso assumes a SHARP interface: this should be << l)")
    return w


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    regime = frozen_regime()
    print(f"frozen colony, R0={regime['R0']:.0f}, {steps} steps, "
          f"l={L_DIFF} units\n")

    traj, snaps = test_rectangle(regime, steps)
    lam = test_mode_decay(regime, steps)
    w = test_interface_width(regime)

    ks = np.array(sorted(lam))
    ls = np.array([lam[k] for k in ks])
    good = np.isfinite(ls)
    ks, ls = ks[good], ls[good]

    print("\n  which law fits the decay?")
    fits = {}
    for name, basis in (('capillary  -k(k^2-1)', ks * (ks ** 2 - 1.0)),
                        ('linear     -k', ks.astype(float)),
                        ('quadratic  -k^2', ks.astype(float) ** 2)):
        c = np.sum(basis * ls) / np.sum(basis * basis)
        resid = ls - c * basis
        r2 = 1.0 - np.sum(resid ** 2) / np.sum((ls - ls.mean()) ** 2)
        fits[name] = (c, r2)
        print(f"    {name:22} coeff {c:+.3e}   r2 = {r2:.4f}")
    cap_c = fits['capillary  -k(k^2-1)'][0]
    Rs = regime['R0'] / L_DIFF
    print(f"\n  if capillary: sigma = -coeff * R*^3 = {-cap_c * Rs**3:.4f} "
          f"(dimensionless, R* = {Rs:.1f})")
    print(f"  (Giverso's Fig.2 spans sigma = 0.007 to 10)")

    fig = plt.figure(figsize=(15, 8.5))
    for i, (step, snap) in enumerate(snaps):
        ax = fig.add_subplot(2, 3, i + 1)
        for p, rr in snap:
            ax.add_patch(Circle(p, rr, color='#43c463', lw=0))
        ax.set_xlim(0, regime['L']); ax.set_ylim(0, regime['L'])
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'rectangle, step {step}', fontsize=10)

    ax = fig.add_subplot(2, 3, 3)
    ax.plot([t for t, _ in traj], [c for _, c in traj], 'o-')
    ax.axhline(1.0, color='k', ls=':', label='disc')
    ax.set_ylim(bottom=0.9)
    ax.set(xlabel='step', ylabel='aspect ratio', title='does the rectangle round off?')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(ks, ls, 'ko-', label='measured decay')
    for name, basis in (('capillary $-k(k^2-1)$', ks * (ks ** 2 - 1.0)),
                        ('linear $-k$', ks.astype(float))):
        c = np.sum(basis * ls) / np.sum(basis * basis)
        ax.plot(ks, c * basis, '--', label=f'{name}  ($r^2$={fits[[n for n in fits if n.split()[0] in name][0]][1]:.3f})'
                if False else f'{name}')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set(xlabel='mode k', ylabel='$\\lambda$ (decay rate)',
           title='Mode decay with growth OFF: capillary ($k^3$) or not?')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Does CellFlow have a Young-Laplace surface tension and a sharp '
                 'interface?', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(HERE, 'giverso_surface_tension.png')
    fig.savefig(p, dpi=110)
    print(f"\nSaved -> {p}")


if __name__ == '__main__':
    main()
