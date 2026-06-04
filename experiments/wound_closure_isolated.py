"""Rigorous wound closure: isolate the hydrodynamic COUPLING RANGE from speed.

The confound in wound_closure_hydro.py: sweeping the screening length delta at
fixed viscosity mu also changes the substrate drag alpha = mu/delta^2, so larger
delta means both longer-range coupling AND lower friction (faster). Here we
remove the speed confound.

Key fact: at fixed delta the Brinkman flow scales exactly as 1/mu, so an
isolated cell's self-mobility is S(delta)/mu. We therefore CALIBRATE
mu(delta) = S(delta) / v0 so that a single cell moves at the same speed v0 for
every delta. Any remaining difference in closure dynamics is then attributable
to the coupling RANGE alone, not to raw mobility.

Metrics: closure time, front roughness (std of the front position), and the
dominant fingering wavelength (peak of the front's spatial power spectrum).

Run:  python experiments/wound_closure_isolated.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation  # noqa: E402
from cellflow.fluid.brinkman_fft import solve_velocity, alpha_from_screening_length  # noqa: E402
from cellflow.fluid.ibm import spread_forces_numba, interpolate_velocity_numba  # noqa: E402

L = 120.0
G = 120
DX = L / G
GAP = 40.0
SPACING = 6.0
STEPS = 400
COVER_THRESH = 0.50
X0, X1 = int((L - GAP) / 2), int((L + GAP) / 2)
F_REF = 30.0          # reference single-cell force magnitude (= max_propulsive_force)
DELTAS = [8.0, 16.0, 32.0, 64.0]


def self_mobility_factor(delta):
    """S(delta): speed of one isolated cell under force F_REF at mu=1."""
    pos = np.array([[L / 2, L / 2]])
    forces = np.array([[F_REF, 0.0]])
    fd = spread_forces_numba(pos, forces, G, G, DX)
    u = solve_velocity(fd, mu=1.0, dx=DX, alpha=alpha_from_screening_length(1.0, delta))
    v = interpolate_velocity_numba(u, pos, DX)
    return float(np.hypot(v[0, 0], v[0, 1]))


def config(mu, delta):
    return {
        'initial_setup_type': 'wound_healing',
        'wound_gap_width': GAP, 'initial_cell_spacing': SPACING,
        'dt': 0.01, 'physical_size': L, 'grid_resolution': G,
        'nutrient_bc_value': 15.0, 'nutrient_D': 0.5, 'chi_nutrient': 12.0,
        'walk_speed': 0.0, 'max_propulsive_force': F_REF, 'viscosity': mu,
        'adhesion_strength': 0.3, 'adhesion_cutoff_factor': 1.5,
        'repulsion_strength': 50.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': delta,
    }


def centerline_ridge(amp=18.0, width=28.0):
    x = np.arange(G) * DX
    return np.tile(amp * np.exp(-((x - L / 2) / width) ** 2), (G, 1))


def coverage_grid(sim):
    cov = np.zeros((G, G), dtype=bool)
    for c in sim.cells:
        cx, cy = c.position
        r = c.radius
        gx, gy = int(cx / DX), int(cy / DX)
        rr = int(np.ceil(r / DX))
        for j in range(max(0, gy - rr), min(G, gy + rr + 1)):
            for i in range(max(0, gx - rr), min(G, gx + rr + 1)):
                if (i * DX - cx) ** 2 + (j * DX - cy) ** 2 <= r * r:
                    cov[j, i] = True
    return cov


def left_front(cov):
    """Rightmost covered x (in the left half) per row -> the advancing front."""
    xmax = int(L / 2)
    h = np.full(G, np.nan)
    for j in range(G):
        covered = np.where(cov[j, :xmax])[0]
        if covered.size:
            h[j] = covered[-1]
    return h


def fingering_wavelength(h):
    """Dominant wavelength of front fluctuations via spatial power spectrum."""
    h = h[~np.isnan(h)]
    if h.size < 8:
        return 0.0, 0.0
    f = h - h.mean()
    ps = np.abs(np.fft.rfft(f)) ** 2
    ps[0] = 0.0
    k = np.argmax(ps)
    if k == 0:
        return 0.0, float(np.std(h))
    wavelength = h.size / k * DX
    return float(wavelength), float(np.std(h))


def run(mu, delta, seed=20240601):
    np.random.seed(seed)
    sim = CellSimulation(config(mu, delta), config_name=f'iso_mu{mu:g}_d{delta:g}')
    ridge = centerline_ridge()
    closure_step = None
    for step in range(STEPS):
        sim.nutrient_field = ridge.copy()
        sim._simulation_step()
        if step % 5 == 0 or step == STEPS - 1:
            wc = coverage_grid(sim)[:, X0:X1].mean()
            if closure_step is None and wc >= COVER_THRESH:
                closure_step = step
                break
    cov = coverage_grid(sim)
    wl, rough = fingering_wavelength(left_front(cov))
    return dict(mu=mu, delta=delta,
                closure_step=closure_step if closure_step is not None else STEPS,
                closed=closure_step is not None, roughness=rough, wavelength=wl,
                n_cells=len(sim.cells))


def main():
    # Calibrate mu(delta) so single-cell speed v0 is constant across delta.
    S = {d: self_mobility_factor(d) for d in DELTAS}
    v0 = np.median(list(S.values()))     # target single-cell speed
    mu_of = {d: S[d] / v0 for d in DELTAS}

    print("Calibration (single-cell speed held fixed):")
    print(f"{'delta':>7} {'S(delta)':>10} {'mu(delta)':>10}")
    for d in DELTAS:
        print(f"{d:>7.0f} {S[d]:>10.4f} {mu_of[d]:>10.4f}")
    # verify constant mobility
    checks = [self_mobility_factor(d) / mu_of[d] for d in DELTAS]
    print(f"calibrated single-cell speed: {np.mean(checks):.4f} "
          f"(spread {100*(max(checks)-min(checks))/np.mean(checks):.2f}%)\n")

    print(f"{'delta':>7} {'mu':>8} {'closure':>8} {'roughness':>10} {'finger_lambda':>14}")
    results = []
    for d in DELTAS:
        r = run(mu_of[d], d)
        results.append(r)
        print(f"{d:>7.0f} {r['mu']:>8.3f} {r['closure_step']:>8d} "
              f"{r['roughness']:>10.2f} {r['wavelength']:>14.1f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    dd = [r['delta'] for r in results]
    ax[0].plot(dd, [r['closure_step'] for r in results], 'o-')
    ax[0].set(xlabel='screening length delta', ylabel='closure time (steps)',
              title='Closure time vs coupling range\n(single-cell speed held FIXED)')
    ax[1].plot(dd, [r['roughness'] for r in results], 's-', color='crimson')
    ax[1].set(xlabel='screening length delta', ylabel='front roughness (std)',
              title='Front fingering vs coupling range')
    ax[2].plot(dd, [r['wavelength'] for r in results], '^-', color='darkgreen')
    ax[2].set(xlabel='screening length delta', ylabel='dominant finger wavelength',
              title='Finger wavelength vs coupling range')
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle("Hydrodynamic coupling range isolated from mobility", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'wound_closure_isolated.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
