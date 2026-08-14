"""What does the colony ACTUALLY look like? -- the morphology behind lambda(k).

`giverso_dispersion.py` measures the dispersion relation and finds no interior
peak: only the longest-wavelength modes (k <~ 3) grow, and they grow slowly. This
script renders what that means, in the least-stabilised regime tested (one overlap
sweep, no growth source), giving fingering its best shot:

  A. free growth  -- no seeded mode, only the model's own cell-scale noise.
     Asks: do fingers emerge spontaneously?
  B. seeded k=3   -- the most unstable mode, at large (nonlinear) amplitude.
     Asks: what does the unstable mode grow INTO?
  C. seeded k=10  -- a finger-like front handed to the model for free.
     Asks: if we simply GIVE it fingers, does it keep them?

C is the sharpest test. Growing fingers from noise is hopeless on time grounds
(the measured rates give ~1.5x mode gain per doubling of colony radius, so ~8
doublings are needed to reach visible lobes). But a model that reproduces
Giverso's physics should at least SUSTAIN fingers once they exist. If seeded
fingers heal back to a disk, the front is stable in the way that matters.

Run:  python experiments/giverso_morphology.py
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
from cellflow.analysis.front import analyze_front                   # noqa: E402
import giverso_dispersion as gd                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Default: least-stabilised regime at the original size (one overlap sweep, no
# growth source, proliferating). Pass 'wide' to instead use the WIDEST unstable
# band reachable here -- short diffusion length AND a larger colony, which the
# measured scaling k0 ~ R/sqrt(a*l) puts near k0 ~ 7.5 (measured).
REGIME = dict(gd.PROLIFERATING, name='morphology',
              overlap_iterations=1, steps=700, sample_every=100)

CASES = [
    ('A: free growth (noise only)', 0, 0.0),
    ('B: seeded k=3, eps=0.30', 3, 0.30),
    ('C: seeded k=10, eps=0.30', 10, 0.30),
]

WIDE = dict(gd.SHORT_L, name='morphology_wide', overlap_iterations=1,
            L=1400.0, G=700, R0=317.0, steps=700, sample_every=100)

# 'prop' -- flux-proportional chemotaxis, the first regime whose dispersion
# relation has a genuine INTERIOR maximum (k* = 3, 10 sigma above its neighbours;
# k0 = 13.5). This is the one that should actually produce shape selection.
PROP = dict(gd.PROPORTIONAL, name='morphology_proportional',
            steps=700, sample_every=100)

if 'wide' in sys.argv[1:]:
    REGIME = WIDE
elif 'propbig' in sys.argv[1:]:
    # 2x radius (~29k cells): the selected mode should scale up with the colony,
    # so this is where a genuinely multi-lobed morphology should appear if the
    # k* ~ R/sqrt(a*l) scaling holds.
    REGIME = dict(gd.PROPORTIONAL_BIG, name='morphology_proportional_2x',
                  steps=1000, sample_every=200)
    CASES = [('A: free growth (noise only)', 0, 0.0),
             ('B: seeded k=6, eps=0.25', 6, 0.25),
             ('C: seeded k=12, eps=0.25', 12, 0.25)]
elif 'prop' in sys.argv[1:]:
    REGIME = PROP

def run_case(mode, eps, seed=1):
    regime = dict(REGIME, seed_eps=eps)
    sim = CellSimulation(gd.make_config(regime, seed),
                         config_name=f'morph_k{mode}')
    rng = np.random.default_rng(seed)
    sim.cells = gd.seeded_colony(regime, max(mode, 1), rng)
    gd.set_phenotype(sim.cells, regime)
    gd.equilibrate_nutrient(sim, regime)
    center = np.array([regime['L'] / 2, regime['L'] / 2])

    def snap():
        return [(c.position.copy(), c.radius, c.active) for c in sim.cells]

    def front():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        return analyze_front(pos, rad, center)

    first, f0 = snap(), front()
    for step in range(regime['steps']):
        gd.set_phenotype(sim.cells, regime)
        sim._simulation_step()
    last, f1 = snap(), front()
    print(f"  mode {mode}: n {len(first)} -> {len(last)}, "
          f"R {f0['mean_radius']:.1f} -> {f1['mean_radius']:.1f}, "
          f"roughness {f0['roughness']:.4f} -> {f1['roughness']:.4f}, "
          f"a_{max(mode,1)} {f0['modes'][max(mode,1)]:.4f} -> "
          f"{f1['modes'][max(mode,1)]:.4f}", flush=True)
    return first, last, f0, f1


def draw(ax, snapshot, L, title):
    for p, r, active in snapshot:
        ax.add_patch(Circle(p, r, color='#43c463' if active else '#1d2b2b', lw=0))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    L = REGIME['L']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    print("rendering colony morphologies (least-stabilised regime):", flush=True)
    for col, (label, mode, eps) in enumerate(CASES):
        first, last, f0, f1 = run_case(mode, eps)
        draw(axes[0, col], first, L, f"{label}\ninitial")
        draw(axes[1, col], last, L,
             f"after {REGIME['steps']} steps  "
             f"(roughness {f0['roughness']:.3f} -> {f1['roughness']:.3f})")

    fig.suptitle("Colony morphology: green = active rim, dark = quiescent core.\n"
                 "Fingering would show as persistent narrow protrusions.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, f"{REGIME['name']}.png")
    fig.savefig(out, dpi=110)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
