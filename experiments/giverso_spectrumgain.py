"""Tracing or amplification? Per-mode gain of the front over its substrate.

A colony growing faster where there is more food looks rough whether or not its
front is unstable. Roughness alone cannot tell the two apart, and the propulsion-
law control already hinted that most of the roughness in the heterogeneous runs is
inherited rather than generated.

The discriminating measurement is a PER-MODE GAIN. Let

    S_sub(k)   = angular spectrum of the nutrient available along each ray,
                 averaged over the annulus the front will sweep through
    S_front(k) = angular spectrum of the final front R(theta)

Passive tracing gives a front displacement proportional to the local food, so
S_front is a rescaled copy of S_sub and the gain G(k) = S_front(k)/S_sub(k) is
FLAT in k. A genuine interfacial instability amplifies its own unstable band, so
G(k) carries a PEAK near k* even though the forcing does not.

Comparing G(k) between the flux-responsive ('proportional') and flux-blind
('saturated') propulsion laws isolates the instability's contribution: the
substrate, the seed and the colony are identical, only the response law differs.

Run:  python experiments/giverso_spectrumgain.py [steps] [n_seeds]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.analysis.front import (main_cluster_mask, front_radii,  # noqa: E402
                                     front_modes)
import giverso_dispersion as gd                                     # noqa: E402
import giverso_heterogeneous as gh                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_BINS = 180
KMAX = 20
L_DIFF = 8.9
NOISE_A = 0.6
ANNULUS = 60.0            # radial band the front sweeps during the run


def substrate_spectrum(sim, center, R0):
    """Angular spectrum of the food available along each ray.

    Averaged over the annulus [R0, R0 + ANNULUS] -- the band the front actually
    grows through, which is what a tracing front would integrate.
    """
    g = sim.grid_resolution
    ys, xs = np.mgrid[0:g, 0:g]
    x = xs * sim.dx - center[0]
    y = ys * sim.dx - center[1]
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    band = (r >= R0) & (r <= R0 + ANNULUS)
    idx = np.clip(((th + np.pi) / (2 * np.pi) * N_BINS).astype(int), 0, N_BINS - 1)

    prof = np.zeros(N_BINS)
    cnt = np.zeros(N_BINS)
    np.add.at(prof, idx[band], sim.nutrient_field[band])
    np.add.at(cnt, idx[band], 1.0)
    prof = np.where(cnt > 0, prof / np.maximum(cnt, 1), prof.mean())
    m = prof.mean()
    return 2.0 * np.abs(np.fft.rfft(prof - m)) / (N_BINS * m)


def run(response, seed, steps, amplitude=NOISE_A, corr=4 * L_DIFF):
    regime = dict(gd.PROPORTIONAL, seed_eps=0.0, name='gain')
    cfg = gd.make_config(regime, seed)
    cfg['propulsion_response'] = response
    sim = CellSimulation(cfg, config_name='gain')
    rng = np.random.default_rng(seed)
    sim.cells = gd.seeded_colony(regime, 1, rng)
    gd.set_phenotype(sim.cells, regime)
    gd.equilibrate_nutrient(sim, regime)
    center = np.array([regime['L'] / 2, regime['L'] / 2])

    if amplitude > 0:
        noise = gh.correlated_field(sim.nutrient_field.shape, sim.dx, corr, rng)
        sim.nutrient_field *= (1.0 + amplitude * noise)
        np.clip(sim.nutrient_field, 0.0, None, out=sim.nutrient_field)

    def front():
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        keep = main_cluster_mask(pos, rad)
        return front_radii(pos[keep], center, N_BINS)

    R0f = front()
    S_sub = substrate_spectrum(sim, center, R0f.mean())
    a0 = front_modes(R0f)

    for _ in range(steps):
        gd.set_phenotype(sim.cells, regime)
        sim._simulation_step()

    R1 = front()
    a1 = front_modes(R1)
    # remove the colony's pre-existing roughness so we measure the response to
    # the substrate, not the packing noise it started with
    resp = np.sqrt(np.clip(a1[:KMAX + 1] ** 2 - a0[:KMAX + 1] ** 2, 0, None))
    gain = resp / np.maximum(S_sub[:KMAX + 1], 1e-12)
    print(f"  {response:<13} seed={seed}  front rough {np.std(R0f)/np.mean(R0f):.4f}"
          f" -> {np.std(R1)/np.mean(R1):.4f}   R {R0f.mean():.0f}->{R1.mean():.0f}",
          flush=True)
    return S_sub[:KMAX + 1], resp, gain


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seeds = list(range(1, n_seeds + 1))

    print(f"per-mode gain, {steps} steps, seeds {seeds}, "
          f"substrate A={NOISE_A}, xi={4*L_DIFF:.0f}\n")
    out = {}
    for response in ('proportional', 'saturated'):
        subs, resps, gains = [], [], []
        for s in seeds:
            a, b, c = run(response, s, steps)
            subs.append(a); resps.append(b); gains.append(c)
        out[response] = (np.array(subs), np.array(resps), np.array(gains))
        print()

    ks = np.arange(KMAX + 1)
    band = slice(2, KMAX + 1)
    print(f"{'k':>3} {'substrate':>11} {'front resp':>11} "
          f"{'gain(prop)':>11} {'gain(sat)':>11} {'ratio':>7}")
    print('-' * 60)
    gp = out['proportional'][2].mean(axis=0)
    gs = out['saturated'][2].mean(axis=0)
    sub = out['proportional'][0].mean(axis=0)
    rsp = out['proportional'][1].mean(axis=0)
    for k in range(2, min(KMAX, 14) + 1):
        print(f"{k:>3} {sub[k]:>11.5f} {rsp[k]:>11.5f} {gp[k]:>11.2f} "
              f"{gs[k]:>11.2f} {gp[k]/gs[k] if gs[k] > 0 else np.nan:>7.2f}")
    print('-' * 60)
    flatness = lambda g: g[band].max() / g[band].mean()
    print(f"gain 'peakiness' (max/mean over k=2..{KMAX}):  "
          f"proportional {flatness(gp):.2f}   saturated {flatness(gs):.2f}")
    print(f"argmax of gain: proportional k={2 + int(np.argmax(gp[band]))}, "
          f"saturated k={2 + int(np.argmax(gs[band]))}")
    print("\nFLAT gain => the front is tracing the substrate.")
    print("PEAKED gain near k* => the instability is amplifying its own band.")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].semilogy(ks[band], sub[band], 'o-', label='substrate $S_{sub}$')
    ax[0].semilogy(ks[band], rsp[band], 's-', label='front response')
    ax[0].set(xlabel='mode k', ylabel='amplitude', title='substrate vs front')
    ax[0].legend(); ax[0].grid(alpha=0.3)

    for name, g, style in (('proportional', gp, 'o-'), ('saturated', gs, 's--')):
        ax[1].plot(ks[band], g[band], style, label=name)
    ax[1].set(xlabel='mode k', ylabel='gain $S_{front}/S_{sub}$',
              title='per-mode gain\n(flat = tracing, peaked = amplification)')
    ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].plot(ks[band], (gp / np.maximum(gs, 1e-12))[band], 'o-', color='crimson')
    ax[2].axhline(1.0, color='k', ls=':')
    ax[2].set(xlabel='mode k', ylabel='gain ratio prop / sat',
              title='instability contribution\n(>1 = flux response helps)')
    ax[2].grid(alpha=0.3)

    fig.suptitle('Is the rough front amplifying the substrate, or tracing it?',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(HERE, 'giverso_spectrumgain.png')
    fig.savefig(p, dpi=115)
    print(f"\nSaved -> {p}")


if __name__ == '__main__':
    main()
