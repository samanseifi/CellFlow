"""Coupling-range effect of viscosity: at FIXED substrate drag alpha, the
Brinkman screening length delta = sqrt(mu / alpha) grows with viscosity, so
higher viscosity spreads hydrodynamic interactions over a LONGER range (more
collective motion) even though absolute speeds fall.

This complements viscosity_sweep.py, which fixed delta to isolate the 1/mu
magnitude scaling. Here we fix alpha and vary mu to isolate the *range*.

Two deterministic probes of the solver (no simulation noise):

  1. Radial decay of the flow from a point force: the distance at which |u|
     falls to 1/e of its near-field value should track delta = sqrt(mu/alpha).

  2. Hydrodynamic entrainment: an active cell at the origin and a passive cell a
     fixed distance away; the entrainment ratio |u_passive| / |u_active|
     increases with mu as the flow reaches farther.

Produces experiments/viscosity_coupling_range.png and a table.
Run:  python experiments/viscosity_coupling_range.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.fluid.brinkman_fft import solve_velocity  # noqa: E402

G = 256
DX = 1.0
L = G * DX
ALPHA = 0.01           # FIXED substrate drag
MUS = [1.0, 4.0, 16.0, 64.0]


def _gaussian_force(sigma=3.0):
    """A smooth +x Gaussian force blob at the center (avoids point-source ringing)."""
    c = G // 2
    ax = np.arange(G) - c
    X, Y = np.meshgrid(ax, ax)
    blob = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    f = np.zeros((G, G, 2))
    f[:, :, 0] = blob
    return f


def solve_profile(mu):
    """Radial |u|(r) along +x for the Gaussian force at fixed alpha."""
    u = solve_velocity(_gaussian_force(), mu=mu, dx=DX, alpha=ALPHA)
    speed = np.hypot(u[:, :, 0], u[:, :, 1])
    c = G // 2
    r = np.arange(0, c)
    prof = speed[c, c:c + c]
    return r, prof


def main():
    sep = 40
    print(f"Fixed alpha = {ALPHA};  delta = sqrt(mu/alpha)")
    print(f"{'mu':>6} {'delta_pred':>11} {'entrain@%d' % sep:>11}")
    profiles = []
    entrain = []
    for mu in MUS:
        r, prof = solve_profile(mu)
        delta = np.sqrt(mu / ALPHA)
        ent = prof[sep] / prof[0]          # |u| at sep / |u| at center
        entrain.append(ent)
        profiles.append((mu, r, prof, delta))
        print(f"{mu:>6.0f} {delta:>11.1f} {ent:>11.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for mu, r, prof, delta in profiles:
        ax[0].plot(r, prof / prof[0], label=f"mu={mu:.0f} (delta={delta:.0f})")
    ax[0].axvline(sep, color='k', ls=':', alpha=0.5)
    ax[0].set(xlim=(0, 120), xlabel='distance r from force center',
              ylabel='|u| / |u|_center  (normalized)',
              title='Flow reaches farther at higher viscosity\n(fixed substrate drag alpha)')
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    ax[1].plot(MUS, entrain, 'o-', color='crimson')
    ax[1].set(xscale='log', xlabel='viscosity mu (log)',
              ylabel=f'entrainment  |u(r={sep})| / |u(0)|',
              title='Hydrodynamic entrainment grows with viscosity\n'
                    '(longer-range collective coupling)')
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'viscosity_coupling_range.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
