"""Hydrodynamic wound closure: how do viscosity and the Brinkman screening
length affect gap-closure dynamics in the wound_healing assay?

Mechanism-exploration study (qualitative trends, not calibrated prediction).
Two sheets of cells flank a wound gap. The bulk consumes nutrient, so the
cell-poor gap stays nutrient-rich; with walk_speed=0 the deterministic
chemotaxis (+ repulsion from the packed sheets) drives cells into the gap.

We sweep (viscosity mu, screening length delta) and measure:
  - closure time: first step the wound region reaches `COVER_THRESH` coverage,
  - front roughness: ragged (fingering) vs smooth advancing front.

Usage:
  python experiments/wound_closure_hydro.py --quick   # one run, sanity check
  python experiments/wound_closure_hydro.py           # full sweep + phase maps
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402

L = 120.0
G = 120
GAP = 40.0
SPACING = 6.0
STEPS = 400
COVER_THRESH = 0.50
X0, X1 = int((L - GAP) / 2), int((L + GAP) / 2)   # wound column range


def config(mu, delta):
    # Migration-driven closure: a persistent nutrient ridge peaked at the wound
    # centerline (imposed each step, see run()) makes chemotaxis pull both
    # sheets inward. walk_speed=0 -> deterministic; the ridge amplitude is kept
    # modest so cells migrate (direction) without fast growth/division. The
    # fluid mobility (~1/mu) and screening length set the closure dynamics.
    return {
        'initial_setup_type': 'wound_healing',
        'wound_gap_width': GAP, 'initial_cell_spacing': SPACING,
        'dt': 0.01, 'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_value': 15.0, 'nutrient_D': 0.5, 'chi_nutrient': 12.0,
        'walk_speed': 0.0, 'max_propulsive_force': 30.0, 'viscosity': mu,
        'adhesion_strength': 0.3, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': delta,
    }


def centerline_ridge(amp=18.0, width=28.0):
    """Nutrient ridge peaked at x = L/2 (uniform in y). Gradient points toward
    the wound center from both sides -> chemotaxis pulls both sheets inward."""
    x = np.arange(G) * (L / G)
    row = amp * np.exp(-((x - L / 2) / width) ** 2)
    return np.tile(row, (G, 1))


def coverage_grid(sim):
    """Boolean grid marking points within a cell radius of some cell."""
    cov = np.zeros((G, G), dtype=bool)
    dx = sim.dx
    for c in sim.cells:
        cx, cy = c.position
        r = c.radius
        gx, gy = int(cx / dx), int(cy / dx)
        rr = int(np.ceil(r / dx))
        for j in range(max(0, gy - rr), min(G, gy + rr + 1)):
            for i in range(max(0, gx - rr), min(G, gx + rr + 1)):
                if (i * dx - cx) ** 2 + (j * dx - cy) ** 2 <= r * r:
                    cov[j, i] = True
    return cov


def wound_coverage(cov):
    return cov[:, X0:X1].mean()


def front_roughness(cov):
    """Std over rows of the left-front position (rightmost covered x in the
    left half + gap). Higher => more fingered/ragged front."""
    fronts = []
    xmax = int((L / 2))
    for j in range(G):
        covered = np.where(cov[j, :xmax])[0]
        if covered.size:
            fronts.append(covered[-1])
    return float(np.std(fronts)) if fronts else 0.0


def run(mu, delta, seed=20240601, record_snaps=None):
    np.random.seed(seed)
    sim = CellSimulation(config(mu, delta), config_name=f'wound_mu{mu:g}_d{delta:g}')
    ridge = centerline_ridge()
    closure_step = None
    cov_series = []
    snaps = {}
    for step in range(STEPS):
        sim.nutrient_field = ridge.copy()    # hold the inward chemotactic drive
        sim._simulation_step()
        if step % 5 == 0 or step == STEPS - 1:
            cov = coverage_grid(sim)
            wc = wound_coverage(cov)
            cov_series.append((step, wc))
            if closure_step is None and wc >= COVER_THRESH:
                closure_step = step
        if record_snaps and step in record_snaps:
            snaps[step] = (np.array([c.position for c in sim.cells]),
                           np.array([c.radius for c in sim.cells]),
                           sim.nutrient_field.copy())
    final_cov = coverage_grid(sim)
    return {
        'mu': mu, 'delta': delta,
        'closure_step': closure_step if closure_step is not None else STEPS,
        'closed': closure_step is not None,
        'roughness': front_roughness(final_cov),
        'n_cells': len(sim.cells),
        'cov_series': cov_series, 'snaps': snaps,
    }


def quick():
    print("QUICK sanity run (mu=10, delta=24)...")
    snaps = [0, STEPS // 2, STEPS - 1]
    r = run(10.0, 24.0, record_snaps=snaps)
    print(f"  closed={r['closed']}  closure_step={r['closure_step']}  "
          f"roughness={r['roughness']:.2f}  N={r['n_cells']}")
    print("  wound coverage over time:")
    for step, wc in r['cov_series'][::8]:
        bar = '#' * int(wc * 40)
        print(f"    step {step:3d}: {wc:.2f} |{bar}")

    fig, axes = plt.subplots(1, len(snaps), figsize=(5 * len(snaps), 5))
    for ax, step in zip(axes, snaps):
        pos, rad, nut = r['snaps'][step]
        ax.imshow(nut, cmap='viridis', origin='lower', extent=[0, L, 0, L], alpha=0.6)
        ax.axvspan(X0, X1, color='red', alpha=0.08)
        for p, rr in zip(pos, rad):
            ax.add_artist(plt.Circle(p, rr, color='white', alpha=0.85))
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}")
    fig.suptitle("Wound closure sanity run (red band = wound region)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wound_quick.png')
    fig.savefig(out, dpi=110)
    print(f"  Saved -> {out}")


def sweep():
    mus = [1.0, 4.0, 16.0]
    deltas = [8.0, 24.0, 72.0]
    closure = np.zeros((len(mus), len(deltas)))
    rough = np.zeros((len(mus), len(deltas)))
    print(f"{'mu':>6} {'delta':>7} {'closure_step':>13} {'roughness':>10} {'N':>5}")
    for a, mu in enumerate(mus):
        for b, delta in enumerate(deltas):
            r = run(mu, delta)
            closure[a, b] = r['closure_step']
            rough[a, b] = r['roughness']
            print(f"{mu:>6.0f} {delta:>7.0f} {r['closure_step']:>13d} "
                  f"{r['roughness']:>10.2f} {r['n_cells']:>5d}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (data, title, cmap) in enumerate([
            (closure, 'Closure time (steps) — lower = faster', 'viridis_r'),
            (rough, 'Front roughness — higher = more fingering', 'magma')]):
        im = ax[k].imshow(data, cmap=cmap, origin='lower', aspect='auto')
        ax[k].set_xticks(range(len(deltas))); ax[k].set_xticklabels([f'{d:.0f}' for d in deltas])
        ax[k].set_yticks(range(len(mus))); ax[k].set_yticklabels([f'{m:.0f}' for m in mus])
        ax[k].set_xlabel('screening length delta'); ax[k].set_ylabel('viscosity mu')
        ax[k].set_title(title)
        for a in range(len(mus)):
            for b in range(len(deltas)):
                ax[k].text(b, a, f'{data[a,b]:.0f}', ha='center', va='center',
                           color='white', fontsize=10)
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle("Hydrodynamic wound closure — phase maps")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wound_closure_phase.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


def compare_fronts():
    """Visualize smooth vs fingered closure: mu=1, short vs long screening."""
    cases = [8.0, 72.0]
    snap_steps = [0, 120, 260]
    fig, axes = plt.subplots(len(cases), len(snap_steps),
                             figsize=(4.5 * len(snap_steps), 4.5 * len(cases)))
    for row, delta in enumerate(cases):
        r = run(1.0, delta, record_snaps=snap_steps)
        for col, step in enumerate(snap_steps):
            ax = axes[row, col]
            pos, rad, _ = r['snaps'][step]
            ax.set_facecolor('#0b1021')
            ax.axvspan(X0, X1, color='red', alpha=0.10)
            for p, rr in zip(pos, rad):
                ax.add_artist(plt.Circle(p, rr, color='deepskyblue', alpha=0.9))
            ax.set_xlim(0, L); ax.set_ylim(0, L)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"delta={delta:.0f}, step={step}", fontsize=10)
        axes[row, 0].set_ylabel(
            f"delta={delta:.0f}\n({'long-range' if delta > 40 else 'short-range'} flow)",
            fontsize=10)
    fig.suptitle("Wound front: short screening -> fingering/stall, "
                 "long screening -> smooth fast closure  (mu=1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wound_fronts.png')
    fig.savefig(out, dpi=110)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--fronts', action='store_true')
    args = ap.parse_args()
    if args.quick:
        quick()
    elif args.fronts:
        compare_fronts()
    else:
        sweep()
