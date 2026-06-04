"""Poroelastic ECM demo (#18): flow reroutes around cell-deposited matrix.

A uniform body force drives fluid left->right. A patch of ECM (as cells would
deposit) raises the local Brinkman drag (lowers permeability). As the matrix
builds, the streamlines bend AROUND the low-permeability region and the flow
through it is choked -- the cell<->fluid feedback that defines this model.

Produces experiments/ecm_poroelastic.png.
Run:  python experiments/ecm_poroelastic_demo.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.fluid.brinkman_fft import solve_velocity_variable_alpha   # noqa: E402

G = 120
L = 120.0
DX = L / G
BASE_ALPHA = 0.4
DRAG = 0.5            # alpha contribution per unit ECM
ECM_LEVELS = [0.0, 4.0, 12.0]    # increasing deposited matrix


def main():
    x = np.arange(G) * DX
    X, Y = np.meshgrid(x, x)
    blob = np.exp(-(((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * 14.0 ** 2)))
    f = np.zeros((G, G, 2))
    f[:, :, 0] = 1.0                  # uniform drive in +x

    fig, axes = plt.subplots(1, len(ECM_LEVELS), figsize=(5 * len(ECM_LEVELS), 5))
    for ax, ecm_amp in zip(axes, ECM_LEVELS):
        ecm = ecm_amp * blob
        alpha = BASE_ALPHA + DRAG * ecm
        u, iters, res = solve_velocity_variable_alpha(f, mu=1.0, dx=DX, alpha_field=alpha)
        speed = np.hypot(u[:, :, 0], u[:, :, 1])

        im = ax.imshow(alpha, cmap='magma', origin='lower', extent=[0, L, 0, L])
        ax.streamplot(x, x, u[:, :, 0], u[:, :, 1], color='cyan', density=1.2,
                      linewidth=0.8, arrowsize=0.8)
        ax.set_title(f"ECM peak={ecm_amp:.0f}  (alpha {alpha.min():.1f}-{alpha.max():.1f}, "
                     f"{iters} iters)\nflow at center = {speed[G//2, G//2]:.3f}",
                     fontsize=9)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Poroelastic ECM: flow reroutes around cell-deposited matrix "
                 "(background = drag alpha; cyan = streamlines)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecm_poroelastic.png')
    fig.savefig(out, dpi=110)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
