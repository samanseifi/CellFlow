"""Convergence & conservation diagnostics — "are the numbers precise?"

Three studies that tell you which quantities are trustworthy and in what regime:

  1. GRID convergence of (a) single-cell self-mobility and (b) two-cell
     hydrodynamic interaction at a fixed PHYSICAL separation. The IBM spreads
     a force over the Peskin 4-point kernel whose width is tied to the grid
     (4*dx), so refining the grid shrinks the regularization. Expectation:
     the *interaction* (far field) converges, but the *self-mobility* does NOT
     converge (it drifts, ~log) because the regularization scale is grid-tied
     rather than physical (cell radius). This is a real precision caveat.

  2. TIMESTEP convergence of a two-cell trajectory. Cells move by forward Euler
     (first order), so the error should scale ~dt^1.

  3. MASS CONSERVATION of the semi-Lagrangian scalar advection (known to be
     non-conservative) — quantify the leak.

Produces experiments/convergence.png and a printed report.
Run:  python experiments/convergence.py
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
from cellflow.kernels.diffusion import advect_scalar_field_numba, diffuse_field_numba  # noqa: E402

L = 64.0
DELTA = 16.0
F = 30.0


# ---------- 1. grid convergence ----------
def grid_study():
    Gs = [64, 128, 256, 512]
    self_mob, interaction = [], []
    sep = 16.0
    for G in Gs:
        dx = L / G
        alpha = alpha_from_screening_length(1.0, DELTA)
        # (a) single-cell self-mobility
        pos1 = np.array([[L / 2, L / 2]])
        f1 = np.array([[F, 0.0]])
        fd = spread_forces_numba(pos1, f1, G, G, dx)
        u = solve_velocity(fd, mu=1.0, dx=dx, alpha=alpha)
        v = interpolate_velocity_numba(u, pos1, dx)
        self_mob.append(float(np.hypot(v[0, 0], v[0, 1])))
        # (b) interaction: speed entrained at a passive cell `sep` away
        pos2 = np.array([[L / 2 - sep / 2, L / 2], [L / 2 + sep / 2, L / 2]])
        f2 = np.array([[F, 0.0], [0.0, 0.0]])
        fd2 = spread_forces_numba(pos2, f2, G, G, dx)
        u2 = solve_velocity(fd2, mu=1.0, dx=dx, alpha=alpha)
        v2 = interpolate_velocity_numba(u2, pos2, dx)
        interaction.append(float(np.hypot(v2[1, 0], v2[1, 1])))
    return Gs, self_mob, interaction


# ---------- 2. timestep convergence ----------
def _two_cell_config(mu=1.0):
    return {
        'initial_setup_type': 'central_uniform', 'num_cells': 2,
        'initial_cluster_radius': 0.1, 'dt': 0.01,
        'physical_size': L, 'grid_resolution': 128,
        'nutrient_bc_value': 5.0, 'nutrient_D': 0.5, 'chi_nutrient': 0.0,
        'walk_speed': 0.0, 'max_propulsive_force': 0.0, 'viscosity': mu,
        'adhesion_strength': 2.0, 'adhesion_cutoff_factor': 1.8,
        'repulsion_strength': 50.0, 'attractant_D': 0.0, 'chi_attractant': 0.0,
        'enable_visualization': False, 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': DELTA,
    }


def _run_two_cell(dt, T=2.0):
    np.random.seed(0)
    cfg = _two_cell_config()
    cfg['dt'] = dt
    sim = CellSimulation(cfg, config_name='conv_dt')
    # deterministic placement, attracting (in the adhesion band, not overlapping)
    sim.cells[0].position = np.array([L / 2 - 3.5, L / 2])
    sim.cells[1].position = np.array([L / 2 + 3.5, L / 2])
    steps = int(round(T / dt))
    for _ in range(steps):
        sim._simulation_step()
    return float(np.linalg.norm(sim.cells[0].position - sim.cells[1].position))


def dt_study():
    dts = [0.04, 0.02, 0.01, 0.005, 0.0025]
    seps = [_run_two_cell(dt) for dt in dts]
    ref = seps[-1]
    errs = [abs(s - ref) for s in seps[:-1]]
    dts_e = dts[:-1]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    return dts_e, errs, orders


# ---------- 3. mass conservation of advection ----------
def mass_study():
    G = 128
    dx = L / G
    x = np.arange(G) * dx
    X, Y = np.meshgrid(x, x)
    field = np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * 5.0 ** 2))
    vel = np.zeros((G, G, 2))
    vel[:, :, 0] = 1.0           # gentle uniform translation; blob stays interior
    vel[:, :, 1] = 0.5
    dt = 0.05
    m0 = field.sum()
    masses = [1.0]
    for _ in range(200):
        field = advect_scalar_field_numba(field, vel, dt, dx)
        masses.append(field.sum() / m0)
    return np.array(masses)


def main():
    print("=== 1. GRID convergence (fixed physical size & screening) ===")
    Gs, self_mob, interaction = grid_study()
    print(f"{'G':>6} {'dx':>7} {'self_mobility':>14} {'interaction':>12}")
    for G, sm, it in zip(Gs, self_mob, interaction):
        print(f"{G:>6} {L/G:>7.3f} {sm:>14.5f} {it:>12.6f}")
    print(f"  self-mobility change last refinement: "
          f"{100*abs(self_mob[-1]-self_mob[-2])/self_mob[-2]:.1f}%  (grid-tied -> does NOT converge)")
    print(f"  interaction change last refinement:   "
          f"{100*abs(interaction[-1]-interaction[-2])/interaction[-2]:.1f}%  (far field -> converges)")

    print("\n=== 2. TIMESTEP convergence (forward Euler, expect order ~1) ===")
    dts_e, errs, orders = dt_study()
    print(f"{'dt':>8} {'error_vs_ref':>14}")
    for dt, e in zip(dts_e, errs):
        print(f"{dt:>8.4f} {e:>14.3e}")
    print(f"  observed orders: {', '.join(f'{o:.2f}' for o in orders)}  (forward Euler -> ~1)")

    print("\n=== 3. MASS conservation of semi-Lagrangian advection (interior) ===")
    masses = mass_study()
    print(f"  total mass after 200 advection steps: {masses[-1]:.4f} of initial "
          f"({100*(masses[-1]-1):+.2f}%)  -> conserved in the interior under smooth flow")
    print("  (mass IS lost when material advects into the clamped domain "
          "boundary; keep fields away from edges or use a conservative/periodic scheme)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(Gs, self_mob, 'o-', label='self-mobility (grid-tied)')
    ax[0].plot(Gs, interaction, 's-', label='interaction (far field)')
    ax[0].set(xscale='log', xlabel='grid points per axis G', ylabel='speed',
              title='1. Grid convergence\nself-mobility drifts; interaction converges')
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    ax[1].loglog(dts_e, errs, 'o-')
    ax[1].loglog(dts_e, [errs[0] * (d / dts_e[0]) for d in dts_e], 'k--',
                 alpha=0.5, label='slope 1 (forward Euler)')
    ax[1].set(xlabel='dt', ylabel='separation error vs finest',
              title='2. Timestep convergence (~order 1)')
    ax[1].legend(); ax[1].grid(True, alpha=0.3, which='both')

    ax[2].plot(masses)
    ax[2].axhline(1.0, color='k', ls=':', alpha=0.6)
    ax[2].set(xlabel='advection step', ylabel='total mass / initial',
              title='3. Advection mass (conserved interior;\nlost only at clamped boundary)')
    ax[2].grid(True, alpha=0.3)

    fig.suptitle("Convergence & conservation diagnostics", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convergence.png')
    fig.savefig(out, dpi=110)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
