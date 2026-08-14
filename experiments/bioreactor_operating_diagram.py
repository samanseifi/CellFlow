"""Bioreactor operating diagram: critical aggregate size vs supply & metabolism.

The companion to bioreactor_aggregate.py. That demo showed, for one medium level,
that a cell aggregate develops a necrotic core past a critical size, and that the
viable shell thickness is set by transport (independent of aggregate size). Here
we map the full design space: the two knobs a bioprocess engineer actually
controls are the **medium supply concentration** (feed/oxygenation) and the
**cell-line metabolic uptake rate**. For each combination we measure the steady
viable-shell thickness and convert it to the **critical aggregate radius** -- the
size at which the colony is 50% necrotic by area.

Geometry: viable fraction (2D) = 1 - (1 - shell/R)^2, so 50% viable occurs at
R_crit = shell / (1 - sqrt(0.5)) = 3.41 * shell. Because the shell is
size-independent, ONE equilibration per (supply, uptake) point fixes R_crit --
making the whole diagram cheap.

Output: bioreactor_operating_diagram.png  (heatmap of R_crit with contours)

Run:  python experiments/bioreactor_operating_diagram.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.bioreactor_aggregate as B                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R_PROBE = 75.0            # aggregate radius used to read the (size-independent) shell
STEPS = 700              # diffusion steps to quasi-steady
HALF = 1.0 - np.sqrt(0.5)  # shell/R at 50% viable  (R_crit = shell / HALF)

SUPPLIES = [30.0, 60.0, 120.0, 240.0, 480.0]      # medium concentration (a.u.)
UPTAKES = [0.4, 0.8, 1.6, 3.2]                     # cell-line metabolic rate


def main():
    pos = B.hex_disk(R_PROBE)
    print(f"probe aggregate R={R_PROBE:.0f} ({len(pos)} cells); "
          f"mapping {len(SUPPLIES)}x{len(UPTAKES)} operating points ...", flush=True)
    Rcrit = np.zeros((len(UPTAKES), len(SUPPLIES)))
    for i, cons in enumerate(UPTAKES):
        B.CONSUMPTION = cons
        for j, bc in enumerate(SUPPLIES):
            field, _, _ = B.equilibrate(pos, bc, STEPS)
            shell = B.shell_thickness(field, pos)
            # if the probe is fully viable, the shell is capped at R_probe -> the
            # true critical size is at least R_probe/HALF (report the bound).
            rc = min(shell, R_PROBE) / HALF
            Rcrit[i, j] = rc
            print(f"  uptake={cons:4.1f}  supply={bc:5.0f}: shell {shell:5.1f}u "
                  f"-> R_crit ~ {rc:5.0f}u", flush=True)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(Rcrit, origin='lower', aspect='auto', cmap='viridis',
                   extent=[0, len(SUPPLIES), 0, len(UPTAKES)])
    # contours of constant critical size
    X, Y = np.meshgrid(np.arange(len(SUPPLIES)) + 0.5, np.arange(len(UPTAKES)) + 0.5)
    cs = ax.contour(X, Y, Rcrit, colors='white', alpha=0.6, linewidths=1)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')
    for i in range(len(UPTAKES)):
        for j in range(len(SUPPLIES)):
            ax.text(j + 0.5, i + 0.5, f'{Rcrit[i, j]:.0f}', ha='center', va='center',
                    color='black', fontsize=9, fontweight='bold')
    ax.set_xticks(np.arange(len(SUPPLIES)) + 0.5); ax.set_xticklabels([f'{s:.0f}' for s in SUPPLIES])
    ax.set_yticks(np.arange(len(UPTAKES)) + 0.5); ax.set_yticklabels([f'{u:.1f}' for u in UPTAKES])
    ax.set_xlabel('medium supply concentration (a.u.)')
    ax.set_ylabel('cell-line metabolic uptake rate')
    ax.set_title('Bioreactor operating diagram: critical aggregate radius (units)\n'
                 'rich medium + low metabolism -> large viable aggregates')
    fig.colorbar(im, ax=ax, label='critical radius R_crit (units, 50% necrotic)')
    fig.tight_layout()
    out = os.path.join(HERE, 'bioreactor_operating_diagram.png')
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  -> {out}\nDone.")


if __name__ == '__main__':
    main()
