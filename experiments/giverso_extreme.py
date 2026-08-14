"""Push the discrete colony into Giverso's MOST-unstable regime and beyond.

Our earlier "decisive" seeded test (experiments/giverso_seed.py) found a STABLE
front -- but it ran at R/l ~ 3.  Giverso's strongly-branched simulations live at
R*/lc ~ 31, Rout/lc ~ 155, minimum surface tension sigma ~ 0.007.  We were a
factor ~10 short on the single most important group: colony radius / diffusion
length.  At R/l ~ 3 the active rim is as thick as the whole colony, so a
protrusion cannot focus nutrient flux and nothing destabilizes.

This script collapses the diffusion length to ~2 cell radii (high consumption +
low nutrient D) so a colony of only ~5-10k cells reaches R/l ~ 25-50 -- the
paper's regime and BEYOND -- with a one-cell-thick active rim, and minimizes the
steric "surface tension" (low repulsion, single overlap pass).  Quiescence
freezes the starved interior; gradient-directed division pushes daughters up the
(outward) nutrient gradient -> Mullins-Sekerka flux-focusing feedback.

We seed a small mode-m ripple (eps small -- we want the INSTABILITY to amplify
it, not impose it) and track whether the front roughens (UNSTABLE -> fingers) or
heals (STABLE).  Outputs initial/final colony (active=green, passive=dark) and
the front power spectrum + roughness vs time.

Run:  python experiments/giverso_extreme.py            # default R/l ~ 25
      python experiments/giverso_extreme.py 50 14      # R/l target ~50, seed mode 14
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
from cellflow.kernels.diffusion import diffuse_field_numba          # noqa: E402
from cellflow.kernels.fields import absorb_nutrient_numba           # noqa: E402

# ---- regime knobs (VERIFIED with experiments/giverso_lcheck.py) -------------
# D=6, cons=0.6, thresh=25 -> ~3 cell-radius active rim, starved quiescent core,
# R/rim ~ 12 (a genuine thin advancing front -- the regime our R/l~3 runs lacked).
L = 400.0                 # box (physical units)
G = 300                   # nutrient grid  -> dx ~ 1.33
NUTRIENT_D = 6.0          # FAST diffusion (quasi-steady, as the paper assumes) ...
CONSUMPTION = 0.6         # ... matched uptake => thin active rim ~3 cell radii
BC_VALUE = 100.0          # strong far-field nutrient reservoir (steep rim gradient)
QUI_THRESH = 25.0         # interior below this goes passive -> thin active rim
# Low conversion efficiency (the paper's small-beta branching regime): cells burn
# most uptake on basal metabolism, so growth = uptake - basal is a SMALL DIFFERENCE
# of large fluxes.  The front then advances slower than nutrient diffuses (truly
# quasi-steady -> thin rim is SUSTAINED) and growth is hypersensitive to flux:
# a tip with slightly more nutrient grows disproportionately faster (Mullins-
# Sekerka), while starved valley cells (uptake < basal) die and carve channels.
BASAL = 44.0              # just below the stall (basal=50 froze): small +net growth
CELL_R = 2.4
STEPS = 70                 # capture the PEAK-instability morphology (relaxes after)
SEED_EPS = 0.06           # SMALL seed: let the instability grow it (or not)


def config():
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 1,
        'initial_cluster_radius': 1.0, 'dt': 0.05,
        'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': BC_VALUE,
        'nutrient_D': NUTRIENT_D, 'chi_nutrient': 0.0,         # no chemotactic smoothing
        'walk_speed': 0.02, 'max_propulsive_force': 1.0,
        'adhesion_strength': 0.0, 'adhesion_cutoff_factor': 1.2,
        'repulsion_strength': 14.0,                            # LOW steric surface tension
        'overlap_iterations': 1,                               # minimal re-rounding
        'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': 500.0, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': 12.0,
        'growth_model': 'area_conserving', 'enable_visualization': False, 'seed': 7,
        'enable_quiescence': True, 'quiescence_nutrient_threshold': QUI_THRESH,
        'directed_division': True,
    }


def measure_l(sim, center):
    """Penetration depth of nutrient into the colony: fit excess ~ exp(-d/l)."""
    field = sim.nutrient_field
    g = G
    ys, xs = np.mgrid[0:g, 0:g]
    cx = int(center[0] / sim.dx); cy = int(center[1] / sim.dx)
    rr = np.hypot(xs - cx, ys - cy) * sim.dx
    # radial profile (median per ring), interior plateau subtracted
    bins = np.arange(0, L / 2, 3.0)
    prof = np.array([np.median(field[(rr >= bins[i]) & (rr < bins[i + 1])])
                     if np.any((rr >= bins[i]) & (rr < bins[i + 1])) else np.nan
                     for i in range(len(bins) - 1)])
    return bins, prof


def lobed_colony(R0, mode, cell_r=CELL_R):
    spacing = 1.9 * cell_r
    Cell.next_id = 0
    cells = []
    c = L / 2
    n = int(2 * R0 * (1 + SEED_EPS) / (spacing * np.sqrt(3) / 2)) + 2
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        nx = int(2 * R0 * (1 + SEED_EPS) / spacing) + 2
        for i in range(-nx, nx + 1):
            x = i * spacing + xoff
            r = np.hypot(x, y); th = np.arctan2(y, x)
            if r <= R0 * (1 + SEED_EPS * np.cos(mode * th)):
                cell = Cell(np.array([c + x, c + y]), nutrient=40.0, area_conserving=True)
                cell.radius = cell_r
                cell.consumption_rate = CONSUMPTION     # force the short-l regime
                cell.basal_metabolism_rate = BASAL      # low conversion efficiency
                cells.append(cell)
    return cells


def front_radii(sim, center, n_bins=360):
    pos = np.array([c.position for c in sim.cells])
    d = pos - center
    th = np.arctan2(d[:, 1], d[:, 0]); rad = np.hypot(d[:, 0], d[:, 1])
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(th, edges) - 1
    R = np.full(n_bins, np.nan)
    for b in range(n_bins):
        rr = rad[idx == b]
        if rr.size:
            R[b] = rr.max()
    good = ~np.isnan(R)
    return np.interp(np.arange(n_bins), np.where(good)[0], R[good], period=n_bins)


def spectrum(R):
    f = np.fft.rfft(R - R.mean())
    return 2.0 * np.abs(f) / len(R)


def roughness(R):
    return np.std(R) / np.mean(R)        # dimensionless front roughness


def main():
    R0 = float(sys.argv[1]) if len(sys.argv) > 1 else 110.0
    seed_mode = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    cfg = config()
    sim = CellSimulation(cfg, config_name='giverso_extreme')
    center = np.array([L / 2, L / 2])

    # regime verified separately (giverso_lcheck.py): D=6,cons=0.6 -> ~3 cell-R
    # active rim, R/rim ~ R0/7.  No l-probe needed here.
    R0 = min(R0, 0.45 * L)
    print(f"colony R0 = {R0:.0f}  (R/rim ~ {R0 / 7.0:.0f}), seed mode {seed_mode}, "
          f"D={NUTRIENT_D}, cons={CONSUMPTION}", flush=True)

    sim.cells = lobed_colony(R0, seed_mode)
    print(f"initial cells: {len(sim.cells)}", flush=True)

    # Pre-equilibrate the nutrient field with the colony FROZEN (no growth), so
    # the starved quiescent interior + thin active rim exist before any division.
    # Otherwise the field starts full everywhere and the whole colony feasts and
    # divides in a burst (~5x) that erases the seed before the rim forms.
    sim.nutrient_field[:] = BC_VALUE
    gy, gx = np.mgrid[0:G, 0:G]
    rr = np.hypot(gx - G / 2.0, gy - G / 2.0) * sim.dx
    sim.nutrient_field[rr < R0 - 20.0] = 0.0      # pre-starve the deep interior
    for _ in range(600):
        sim.nutrient_field = diffuse_field_numba(sim.nutrient_field, NUTRIENT_D,
                                                 sim.dt, sim.dx, BC_VALUE)
        read = sim.nutrient_field.copy()
        for cell in sim.cells:
            absorb_nutrient_numba(cell.position, cell.radius, sim.nutrient_field,
                                  read, sim.dt, CONSUMPTION, sim.dx)
    nactive = sum(1 for cell in sim.cells
                  if sim.nutrient_field[int(cell.position[1] / sim.dx),
                                        int(cell.position[0] / sim.dx)] > QUI_THRESH)
    print(f"after field pre-equilibration: {nactive}/{len(sim.cells)} cells in "
          f"active rim ({100*nactive/len(sim.cells):.0f}%)", flush=True)
    snap0 = [(c.position.copy(), c.radius, c.active) for c in sim.cells]
    R = front_radii(sim, center)
    rough0 = roughness(R)
    amp0 = spectrum(R)[seed_mode] / R.mean()

    times, roughs, amps, ncells = [0], [rough0], [amp0], [len(sim.cells)]
    for step in range(STEPS):
        for cell in sim.cells:          # daughters inherit the regime, not defaults
            cell.consumption_rate = CONSUMPTION
            cell.basal_metabolism_rate = BASAL
        sim._simulation_step()
        if step % 10 == 0:
            R = front_radii(sim, center)
            times.append(step + 1)
            roughs.append(roughness(R))
            amps.append(spectrum(R)[seed_mode] / R.mean())
            ncells.append(len(sim.cells))
            if step % 50 == 0:
                print(f"  step {step:4d}: {len(sim.cells):5d} cells, "
                      f"roughness {roughs[-1]/roughs[0]:.2f}x, "
                      f"mode-{seed_mode} {amps[-1]/amps[0]:.2f}x", flush=True)

    Rf = front_radii(sim, center)
    sp = spectrum(Rf) / Rf.mean()
    verdict = ("ROUGHENS -> front UNSTABLE (fingering!)" if roughs[-1] > 1.6 * roughs[0]
               else "stays smooth -> front STABLE")

    fig, ax = plt.subplots(2, 2, figsize=(15, 13))
    for a, (snap, ttl) in zip([ax[0, 0], ax[0, 1]], [
            (snap0, f"seeded R/rim~{R0/7.0:.0f}, mode {seed_mode}, eps={SEED_EPS}"),
            ([(c.position, c.radius, c.active) for c in sim.cells],
             f"after {STEPS} steps ({len(sim.cells)} cells)")]):
        for p, r, act in snap:
            a.add_patch(Circle(p, r, color='limegreen' if act else '#1d2b2b', lw=0))
        a.set_xlim(0, L); a.set_ylim(0, L); a.set_aspect('equal')
        a.set_xticks([]); a.set_yticks([]); a.set_title(ttl)

    ax[1, 0].plot(times, np.array(roughs) / roughs[0], 'o-', label='roughness / initial')
    ax[1, 0].plot(times, np.array(amps) / amps[0], 's-', alpha=0.7,
                  label=f'mode-{seed_mode} amp / initial')
    ax[1, 0].axhline(1.0, color='k', ls=':', alpha=0.6)
    ax[1, 0].set(xlabel='step', ylabel='relative to initial', title='front growth/decay')
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)

    ks = np.arange(1, min(40, len(sp)))
    ax[1, 1].bar(ks, sp[1:len(ks) + 1])
    ax[1, 1].set(xlabel='angular mode k', ylabel='amplitude / R',
                 title='final front power spectrum')
    ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Giverso EXTREME regime: {verdict}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'giverso_extreme.png')
    fig.savefig(out, dpi=110)
    print(f"roughness: {roughs[0]:.4f} -> {roughs[-1]:.4f}  ({roughs[-1]/roughs[0]:.2f}x)")
    print(f"mode-{seed_mode} amp/R: {amps[0]:.4f} -> {amps[-1]:.4f}  ({amps[-1]/amps[0]:.2f}x)")
    print(f"dominant final mode: k={1 + int(np.argmax(sp[1:40]))}")
    print(verdict)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
