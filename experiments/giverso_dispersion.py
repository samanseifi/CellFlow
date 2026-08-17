"""Measure the DISPERSION RELATION lambda(k) of a growing colony front.

This replaces the ad-hoc "does it look rough at the end?" test that produced a
retracted result (see docs/giverso_replication.md). Giverso et al. (2015) predict
a band of unstable angular modes with an interior maximum at k*; the honest way
to ask whether CellFlow reproduces that is to seed each mode separately and
measure its exponential growth rate.

Protocol, per (mode k, seed):
  1. Build a colony with front R(theta) = R0 (1 + eps cos(k theta)), eps small,
     on a JITTERED hex packing (an exact lattice imprints spurious k = 6, 12, 18
     harmonics on the spectrum).
  2. Let the nutrient field relax to quasi-steady with the colony frozen, so the
     starved core / active rim structure exists before any growth. The colony is
     allowed to set up its own profile -- nothing is imposed.
  3. Run, sampling the front on the MAIN CONNECTED CLUSTER only.
  4. Fit lambda from log(amplitude) vs time over the linear regime.

Guards -- a run is only 'valid' if all of them pass:
  * connected-cluster filter        -- detached cells cannot fake roughness
  * advancing-front assertion       -- M-S is an instability of a GROWING front;
                                       a receding colony is disqualified
  * equilibration convergence       -- the quasi-steady field reached a fixed point
  * amplitude floor                 -- a mode below ~one cell radius of lobe
                                       amplitude is not resolvable
  * advective CFL < 1               -- above this the field advection is unstable
  * exponential fit quality (r^2)   -- a transient excursion is not growth

Verdict convention: the instability criterion is growth of the RELATIVE
amplitude a_k = delta_k / R (the shape deviating more and more). lambda_abs, for
the absolute amplitude delta_k, is reported alongside; on a colony expanding at
rate d(lnR)/dt they differ by exactly that rate.

Usage
-----
  python experiments/giverso_dispersion.py calibrate     # known-stable R/l~3 case
  python experiments/giverso_dispersion.py stage1        # current physics
  python experiments/giverso_dispersion.py stage3        # + growth-driven Darcy
  python experiments/giverso_dispersion.py prolif        # front advances by DIVISION
  python experiments/giverso_dispersion.py probe [prolif]  # regime check, 1 run
  python experiments/giverso_dispersion.py plot <name>   # one dispersion relation
  python experiments/giverso_dispersion.py compare [names...]   # overlay

Results are written as JSON next to this file.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellflow.simulation import CellSimulation                      # noqa: E402
from cellflow.cell import Cell                                      # noqa: E402
from cellflow.analysis.front import analyze_front, fit_growth_rate  # noqa: E402
from cellflow.kernels.diffusion import (                            # noqa: E402
    diffuse_field_numba, diffuse_field_implicit_numba)
from cellflow.kernels.fields import absorb_nutrient_numba           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from giverso_analytic import n0_of                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------
# 'CALIBRATION' reproduces the setting of experiments/giverso_seed.py, where the
# front is KNOWN to be strongly stable (a seeded mode decayed ~370x). The harness
# must return lambda < 0 there or it cannot be trusted anywhere else.
CALIBRATION = dict(
    name='calibration_Rl3',
    L=600.0, G=300, R0=90.0, cell_r=2.4,
    nutrient_D=0.02, consumption=0.2, basal=0.0, bc_value=40.0,
    qui_threshold=28.0, screening=15.0, viscosity=500.0,
    adhesion=0.0, repulsion=35.0, overlap_iterations=1,
    walk_speed=0.05, max_force=2.0,
    diffusion_solver='explicit', dt=0.05,
    # Matches experiments/giverso_seed.py exactly (700 steps, eps=0.35), where
    # the seeded mode is known to decay ~370x. A shorter/smaller-amplitude run
    # only moves the amplitude a few percent and cannot discriminate.
    steps=700, sample_every=25, seed_eps=0.35,
)

# Stage 1: the regime the earlier work argued for (thin active rim, R/l >> 1),
# but with the confounders removed --
#   death OFF   : basal = 0, so no cell can starve to death. Giverso's model has
#                 no death; in the retracted run, death was doing the work.
#   adhesion ON : cohesion, so the rim cannot shed and fake a rough front.
#   quiescence  : the legitimate sharp-rim mechanism (starved core freezes).
# Scale separation targets l/a ~ 6-10 and R/l ~ 12-15.
#
# The (D, consumption, R0) values are not guesses: a converged scan of the
# quasi-steady profile (see measure_length_scale) gives l ~ 19 units ~ 8 cell
# radii here, with R/l ~ 10 and an active rim ~10 cell radii thick (~18% of the
# colony). Pushing l/a higher at fixed R lowers R/l and vice versa -- the two
# requirements trade off, and this is the best joint compromise at a colony size
# that still runs in ~1 min.
STAGE1 = dict(
    name='stage1_current_physics',
    L=600.0, G=300, R0=200.0, cell_r=2.4,
    nutrient_D=5.0, consumption=0.02, basal=0.0, bc_value=100.0,
    qui_threshold=30.0, screening=14.0, viscosity=500.0,
    adhesion=0.5, repulsion=35.0, overlap_iterations=2,
    walk_speed=0.02, max_force=1.0,
    diffusion_solver='implicit', dt=0.05,
    # seed_eps must stay well above the cell-scale amplitude floor a/R ~ 0.012
    # (below ~one cell radius of lobe amplitude the measured rate flattens and
    # understates the truth -- see tests/test_front_analysis.py). 0.08 leaves a
    # factor ~7 of headroom while still being a small (linear) perturbation.
    steps=200, sample_every=10, seed_eps=0.08,
)

# Stage 3: identical, plus growth-driven expansion (div u = s). Overlap sweeps
# are reduced because the pressure field now carries part of the expansion --
# see the double-counting note in CellSimulation.
STAGE3 = dict(STAGE1, name='stage3_darcy_expansion',
              overlap_iterations=1,
              growth_source=True, growth_source_strength=1.0)


# Proliferating variant. In STAGE1 the colony advances by cells SWELLING -- the
# cell count is constant, because a cell needs to take up n = 50 -> 100 to divide
# and at consumption = 0.02 that is ~25 time units, longer than the whole run.
# Here cells start with an asynchronous cell-cycle phase and the run is long
# enough to cover >1 division cycle, so the front advances by PROLIFERATION.
# Smaller cells (max_radius 2.5 rather than 4.0) keep the front well resolved and
# the amplitude floor a/R low; the larger box leaves room for the colony to grow.
PROLIFERATING = dict(
    STAGE1, name='stage1_proliferating',
    L=800.0, G=400, R0=180.0,
    max_radius=2.5, cell_r=2.16,      # cell_r = mean radius over one cycle
    async_phase=True,
    steps=700, sample_every=25, seed_eps=0.10,
)


# Scaling-law test. The measured dispersion relations give a marginal mode
# k0 ~ sqrt(R/d0) with d0 an effective capillary length set by cell-scale
# sterics. If that is right, DOUBLING the colony radius at fixed cell size must
# move k0 by sqrt(2) (3.21 -> 4.57) while leaving d0 = R/k0^2 unchanged. If d0
# instead drifts, the scaling law -- and the extrapolation to the ~10^5-10^6
# cells needed for visible fingering -- is wrong.
#
# Note R/l roughly doubles too (9.7 -> 19.7), since l is set by D and uptake, not
# by R; that is closer to the paper's regime but means the destabilising side of
# the balance is not held fixed. Interpret d0 invariance accordingly.
SCALE2 = dict(PROLIFERATING, name='prolif_scale2',
              L=1600.0, G=800, R0=360.0)


# Disentangling control for SCALE2. Going 1x -> 2x in radius changed TWO groups
# at once: R/a (colony vs cell) and R/l (colony vs nutrient penetration depth),
# because l is set by D and uptake, not by R. SCALE2 found k0 ~ R^1.13, far
# stronger than the sqrt(R/d0) surface-tension law -- but which group did it?
#
# Here R and the cell size are held at the 1x values and l is HALVED instead
# (lower D), so R/l matches the 2x run while R/a does not. If k0 jumps to ~7 the
# driver is R/l, and a wider unstable band is reachable at small colony size by
# shortening the diffusion length -- far cheaper than growing the colony. If k0
# stays ~3.2 the driver is R/a, and only bigger colonies will do.
SHORT_L = dict(PROLIFERATING, name='prolif_short_l', nutrient_D=1.25)


# Widest unstable band reachable here: short diffusion length AND a larger
# colony. The two-point power law fitted to the baseline/short-l/2x runs
# extrapolates k0 ~ 10 here -- but a seeded k=10 front still decays in this
# regime, so that extrapolation is not to be trusted. This sweep measures k0
# directly instead of inferring it.
WIDE = dict(SHORT_L, name='prolif_wide', L=1400.0, G=700, R0=317.0)


# Chemotactic expansion -- Giverso's OTHER model ("volumetric versus chemotactic
# expansion" is the paper's title). Every regime above ran with chi_nutrient = 0
# to isolate growth-driven expansion, but the flux diagnostic
# (giverso_fluxresponse.py) showed that leaves the front with NO local response:
# elasticity E ~ 0.03 where Mullins-Sekerka needs ~1. Chemotaxis supplies exactly
# that response -- a tip sees a steeper gradient and migrates outward faster.
# Measured elasticity rises monotonically with chi (0.027 -> 0.063 -> 0.132 ->
# 0.289 for chi = 0, 5, 20, 60) and at chi = 60 a seeded k=10 mode flips from
# decaying to growing. This sweep asks the decisive question: does lambda(k) now
# have an INTERIOR maximum, i.e. a selected wavelength?
CHEMO = dict(SHORT_L, name='chemotactic', overlap_iterations=1,
             chi_nutrient=60.0, max_force=40.0)


# Flux-proportional chemotaxis. The propulsion kernel originally normalised the
# drive and rescaled it to max_propulsive_force, so every cell pushed equally hard
# and chi set only the direction -- which makes a flux-responsive front velocity
# impossible by construction. With propulsion_response='proportional' the
# magnitude is |chi*grad(c)| (capped), and the measured flux elasticity rises
# 0.093 -> 0.327 -> 0.793 -> 1.745 for chi = 5, 20, 60, 150, crossing the E ~ 1
# that Mullins-Sekerka needs. A seeded k=10 mode grows 1.33x. This sweep asks
# whether lambda(k) now has an INTERIOR maximum -- a selected wavelength.
PROPORTIONAL = dict(SHORT_L, name='proportional_chemo', overlap_iterations=1,
                    chi_nutrient=150.0, max_force=2000.0,
                    propulsion_response='proportional')


# The proportional regime scaled up 2x in radius. k* should track the marginal
# mode, which scales as k0 ~ R/sqrt(a*l), so doubling R should roughly double the
# selected mode (k* = 3 -> ~6). This is the test that turns "wavelength selection
# exists" into "the colony has more lobes".
PROPORTIONAL_BIG = dict(PROPORTIONAL, name='proportional_2x',
                        L=1600.0, G=800, R0=360.0)


# The proportional regime with the exterior nutrient field actually equilibrated.
# See the note in equilibrate_nutrient: the default seed leaves the whole medium
# at the reservoir value, and the annulus outside the colony cannot relax within
# a feasible run ((L/2 - R0)^2 / D ~ 39,000 time units here), so the colony edge
# stays at 0.66 n_bc rather than the quasi-steady 0.05. The quiescence threshold
# has to come down with it: it is an ABSOLUTE nutrient level (30 out of 100), so
# on a correctly-drained field every cell would be below it and the colony would
# freeze. Holding it at the same FRACTION of the interface value keeps the
# active-rim structure comparable, which is what makes this a controlled change
# to the nutrient field rather than a change of two things at once.
PROPORTIONAL_FIXED = dict(PROPORTIONAL, name='proportional_fixed_exterior',
                          exterior_quasi_steady=True, qui_threshold=1.4)


# Growth-rate series: IS THE LINEAR-IN-k DAMPING GENERATED BY EXPANSION ITSELF?
#
# Every sweep in this study fits lambda(k) = lambda0 - c*k (r^2 0.94-1.00), and
# the origin of that -c*k has never been identified -- it is the term that pins
# the most-unstable mode at k = 2-3 instead of the 10-30 branching needs.
#
# Two facts point at expansion as its source. Across 12 sweeps c correlates with
# d(lnR)/dt (Pearson +0.62) at a fairly steady ratio c/(dlnR/dt) ~ 0.45. And in
# the surface-tension run, where growth is switched OFF entirely, lambda(k) has
# NO k-dependence at all (+0.0004..-0.0006 over k = 2-10, every candidate law
# fitting equally badly because there is nothing to fit).
#
# That is suggestive but confounded: those 12 sweeps also differ in l, R, the
# propulsion law and overlap handling. This series is the controlled version.
# ``basal`` (basal metabolism) subtracts from a cell's stored nutrient AFTER
# uptake, so it slows net area production while leaving uptake -- and hence the
# nutrient field, l, and the whole spatial structure -- untouched. It is the only
# knob in the model that moves the expansion rate alone.
#
# Prediction if expansion generates the damping: c falls in proportion to
# d(lnR)/dt across the series, at fixed lambda0-per-unit-expansion. If c is
# instead a material property of the mechanics, it stays put while the front
# slows.
#
# Keep basal well below the uptake rate or rim cells starve to death (the biology
# kernel kills a cell whose stored nutrient goes negative), which would reintroduce
# the shedding artifact that invalidated giverso_extreme.py.
GROWTHRATE = [
    dict(PROPORTIONAL, name='growthrate_basal0.00', basal=0.000),
    dict(PROPORTIONAL, name='growthrate_basal0.06', basal=0.060),
    dict(PROPORTIONAL, name='growthrate_basal0.12', basal=0.120),
]


# Screening-length series: IS THE MARGINAL MODE SET BY THE HYDRODYNAMICS?
#
# The Brinkman solver's own transfer function is
#     u_hat(k) = P(k) f_hat(k) / (mu |k|^2 + alpha),   alpha = mu / delta^2
# which is a LOW-PASS FILTER on force. Forces at wavenumber k produce velocity
# suppressed by (mu k^2 + alpha)/alpha relative to k -> 0. At delta = 14 and
# mu = 500 that is a factor of 416 at the cell scale and 47 at the scale of a
# few cells -- which is why a per-cell random force of magnitude 150 produces
# EXACTLY the same displacement as no force at all (measured), and why the
# square never rounds: a corner is sharp on a scale of ~3 cells, precisely
# where the filter removes 98% of the response.
#
# The cutoff sits at mu k^2 = alpha, i.e. k = 1/delta, i.e. colony mode
# m = R/delta. For the proportional regime that is 177/14 = 12.6 against a
# MEASURED marginal mode of 13.45 -- suspiciously close.
#
# This series is the test. delta is varied at fixed R, so R/delta moves by 4x.
# If k0 tracks R/delta, the marginal mode -- and with it the linear-in-k damping
# that has been unexplained since the dispersion work began -- is the
# hydrodynamic coupling, not the cell mechanics.
SCREENING = [
    dict(PROPORTIONAL, name='screening_07', screening=7.0),
    dict(PROPORTIONAL, name='screening_14', screening=14.0),
    dict(PROPORTIONAL, name='screening_28', screening=28.0),
]


# Surface-tension series (issue #35/#36). The falsifiable test of the central
# result: every existing term in this model scales with the expansion rate, so
# lambda(k) = (dlnR/dt) * f(k) and the marginal mode k0 = lam0/c is a pure number
# that no growth-side knob can move. Surface tension is the FIRST term that does
# not scale with growth, so with it active:
#
#   * lam0/c must STOP being constant across a growth-rate series, and
#   * lambda(k) should acquire the Mullins-Sekerka -Gamma k^3 shape, which it has
#     never had (it fits lam0 - c k at r^2 = 0.94-1.00).
#
# sigma enters as a force sigma*kappa ~ sigma/R on boundary cells, against a
# median propulsive force of ~142 in this regime, so the scale that bites is
# sigma ~ 10^2-10^3. Scanned rather than assumed.
SIGMA_SCAN = [dict(PROPORTIONAL, name=f'sigma_{int(sg)}', surface_tension=sg)
              for sg in (0.0, 300.0, 3000.0)]

# Higher sigma, more modes. The first scan showed the Mullins-Sekerka fit quality
# climbing monotonically with sigma (r^2 0.41 -> 0.49 -> 0.75) while k=20 damped
# 2.4x and k=2 barely moved -- the k^3 signature. Four modes cannot discriminate
# two 2-parameter fits, so this resolves the shape properly.
SIGMA_HIGH = [dict(PROPORTIONAL, name=f'sigmahi_{int(sg)}', surface_tension=sg)
              for sg in (3000.0, 10000.0, 30000.0)]

# Growth-rate series repeated WITH surface tension, the controlled comparison
# against GROWTHRATE above.
GROWTHRATE_ST = [
    dict(PROPORTIONAL, name=f'st_growthrate_basal{b:.2f}', basal=b,
         surface_tension=3000.0)
    for b in (0.0, 0.06, 0.12)
]


def make_config(regime, seed):
    cfg = {
        'initial_setup_type': 'central_uniform', 'num_cells': 1,
        'initial_cluster_radius': 1.0, 'dt': regime['dt'],
        'physical_size': regime['L'], 'grid_resolution': regime['G'],
        'nutrient_bc_type': 'dirichlet', 'nutrient_bc_value': regime['bc_value'],
        'nutrient_D': regime['nutrient_D'],
        'chi_nutrient': regime.get('chi_nutrient', 0.0),
        'walk_speed': regime['walk_speed'],
        'max_propulsive_force': regime['max_force'],
        'adhesion_strength': regime['adhesion'], 'adhesion_cutoff_factor': 1.4,
        'repulsion_strength': regime['repulsion'],
        'overlap_iterations': regime['overlap_iterations'],
        'attractant_D': 0.0, 'chi_attractant': 0.0,
        'viscosity': regime['viscosity'], 'fluid_model': 'brinkman_fft',
        'brinkman_screening_length': regime['screening'],
        'growth_model': 'area_conserving', 'enable_visualization': False,
        'seed': seed,
        'enable_quiescence': True,
        'quiescence_nutrient_threshold': regime['qui_threshold'],
        'directed_division': True,
        'diffusion_solver': regime['diffusion_solver'],
        'propulsion_response': regime.get('propulsion_response', 'saturated'),
        'surface_tension': regime.get('surface_tension', 0.0),
        'surface_tension_kmax': regime.get('surface_tension_kmax', 20),
    }
    if regime.get('growth_source'):
        cfg['enable_growth_source'] = True
        cfg['growth_source_strength'] = regime.get('growth_source_strength', 1.0)
    return cfg


# ---------------------------------------------------------------------------
# Colony construction
# ---------------------------------------------------------------------------
def seeded_colony(regime, mode, rng):
    """Jittered hex-packed colony with front R(theta) = R0 (1 + eps cos(k theta)).

    The jitter matters: on an exact hex lattice the front spectrum carries strong
    k = 6, 12, 18, 30 harmonics (the lattice, not physics), which is one of the
    signs the retracted result was an artifact.

    Cell-cycle phase. With ``async_phase`` set, each cell's stored nutrient is
    drawn uniformly over ONE division cycle instead of all cells starting
    identical. Under area-conserving growth a cell divides at n = 100 and each
    daughter restarts at n = 50, so U(50, 100) is exactly the phase distribution
    of a population in steady asynchronous growth. This matters because with
    identical cells the colony divides in synchronized bursts -- which is itself
    a shot-noise event, not sustained proliferation -- and between bursts the
    front advances only by cells swelling.
    """
    R0, cell_r, eps = regime['R0'], regime['cell_r'], regime['seed_eps']
    spacing = 1.9 * cell_r
    jitter = 0.18 * cell_r
    async_phase = bool(regime.get('async_phase', False))
    Cell.next_id = 0
    cells = []
    c = regime['L'] / 2
    n = int(2 * R0 * (1 + eps) / (spacing * np.sqrt(3) / 2)) + 2
    nx = int(2 * R0 * (1 + eps) / spacing) + 2
    for j in range(-n, n + 1):
        y0 = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * spacing + xoff + rng.normal(0.0, jitter)
            y = y0 + rng.normal(0.0, jitter)
            r, th = np.hypot(x, y), np.arctan2(y, x)
            if r <= R0 * (1.0 + eps * np.cos(mode * th)):
                cell = Cell(np.array([c + x, c + y]), area_conserving=True)
                # Stored nutrient MUST be consistent with the radius. Setting
                # them independently makes the first biology step reconcile them
                # with an instantaneous area jump -- a ~300x spike in the growth
                # source, which drives the advection past its CFL limit on step 1.
                # Area-conserving:  A = pi R_max^2 (n/100)  =>  r = R_max sqrt(n/100).
                if 'max_radius' in regime:
                    cell.max_radius = float(regime['max_radius'])
                    cell.min_radius = 0.5 * cell.max_radius
                if async_phase:
                    nut = rng.uniform(50.0, 100.0)
                else:
                    nut = 100.0 * (cell_r / cell.max_radius) ** 2
                cell.nutrient_accumulated = nut
                cell.radius = cell.max_radius * np.sqrt(nut / 100.0)
                cells.append(cell)
    return cells


def set_phenotype(cells, regime):
    """Force the regime's uptake/metabolism on every cell.

    Cell.divide inherits uptake_saturation and polarity but NOT consumption_rate,
    basal_metabolism_rate, or the size limits -- daughters get a fresh
    np.random.normal(0.2, 0.05) and the default min/max radius. Without this call
    the colony drifts out of its regime as it grows.

    Note the batched biology kernel reads min_radius/max_radius from cells[0]
    only, so these must be uniform across the population to mean anything;
    re-applying them every step is what keeps that true after divisions.
    """
    max_r = regime.get('max_radius')
    for cell in cells:
        cell.consumption_rate = regime['consumption']
        cell.basal_metabolism_rate = regime['basal']
        if max_r is not None and cell.max_radius != max_r:
            cell.max_radius = float(max_r)
            cell.min_radius = 0.5 * cell.max_radius
            cell.update_radius()      # daughters are born at the default size


def equilibrate_nutrient(sim, regime, tol=2e-4, max_iter=6000):
    """Relax the nutrient field to the QUASI-STEADY profile, colony frozen.

    Two things matter here, both of which a naive relaxation gets wrong.

    * Starting from a flat field, a colony of radius R takes ~R^2/D for its
      interior to drain -- thousands of time units for these regimes. A short
      relaxation leaves the interior artificially fed, and the measured "active
      fraction" then flips between 0% and 100% as a function of how long you
      happened to run, not of the physics. So we start from the expected
      exponential profile c = c_bc exp(-(R0 - r)/l) and relax from there, which
      begins near the answer instead of far from it.
    * We then iterate to an actual FIXED POINT and report whether it converged,
      rather than trusting a fixed iteration count.

    Nothing is imposed: the initial guess is only a starting point, and the
    converged profile is whatever the colony's own uptake and diffusion produce.
    """
    diffuse = (diffuse_field_implicit_numba
               if regime['diffusion_solver'] == 'implicit' else diffuse_field_numba)
    g = sim.grid_resolution
    c = regime['L'] / 2
    yy, xx = np.mgrid[0:g, 0:g]
    rr = np.hypot(xx * sim.dx - c, yy * sim.dx - c)
    l_guess = max(3.0, np.sqrt(regime['nutrient_D'] / max(regime['consumption'], 1e-9)))
    R0 = regime['R0']

    if regime.get('exterior_quasi_steady', False):
        # The guess above leaves the WHOLE EXTERIOR at the reservoir value, so
        # the colony edge starts at n = n_bc and the medium carries no depletion
        # shadow. That is not a neutral starting point here: equilibrating the
        # annulus from R0 out to the wall takes ~(L/2 - R0)^2 / D time units --
        # 39,000 for the proportional regime, i.e. ~780,000 iterations -- so the
        # relaxation below cannot move it and the guess is effectively IMPOSED.
        # Measured consequence: n at the colony edge stays at 0.69 n_bc and the
        # shadow is gone by 1.2 R, where the quasi-steady solution puts the edge
        # at 0.057 n_bc with the draw-down spanning the dish.
        #
        # So seed the paper's own quasi-stationary state (their Eq. 12) instead:
        # exponential decay inward from the interface value n0, logarithmic
        # recovery outward to the wall.
        Rs = R0 / l_guess
        Rout = (regime['L'] / 2.0) / l_guess
        n0 = float(n0_of(Rs, Rout))
        inner = n0 * np.exp(-np.clip(R0 - rr, 0.0, None) / l_guess)
        with np.errstate(divide='ignore', invalid='ignore'):
            outer = n0 + (1.0 - n0) * np.log(np.maximum(rr, R0) / R0) / np.log(
                (regime['L'] / 2.0) / R0)
        sim.nutrient_field[:] = regime['bc_value'] * np.where(rr < R0, inner,
                                                              np.clip(outer, 0.0, 1.0))
    else:
        sim.nutrient_field[:] = regime['bc_value'] * np.exp(
            -np.clip(R0 - rr, 0.0, None) / l_guess)

    for it in range(max_iter):
        prev = sim.nutrient_field.copy()
        sim.nutrient_field = diffuse(sim.nutrient_field, regime['nutrient_D'],
                                     sim.dt, sim.dx, regime['bc_value'])
        read = sim.nutrient_field.copy()
        for cell in sim.cells:
            absorb_nutrient_numba(cell.position, cell.radius, sim.nutrient_field,
                                  read, sim.dt, regime['consumption'], sim.dx)
        if it % 25 == 24:
            change = np.max(np.abs(sim.nutrient_field - prev)) / regime['bc_value']
            if change < tol:
                return it + 1, True
    return max_iter, False


def check_exterior_steady(sim, regime, R, radii_frac=(1.10, 1.25, 1.45)):
    """Is the field OUTSIDE the colony actually at steady state?

    The existing convergence test asks whether the field stopped changing to a
    tolerance. That is necessary but not sufficient, and it passes on a field
    that is still filling in slowly: at tol = 2e-4 the proportional regime
    reports converged, and at tol = 1e-6 it fails to converge in 40,000
    iterations.

    This is the sufficient test. Outside the colony there are no cells and hence
    no sinks, so conservation makes the radial flux through any circle

        Phi(r) = -2 pi r D dn/dr

    INDEPENDENT of r. Measured spread across radii is therefore a direct measure
    of how far the exterior is from steady state -- no tolerance to pick, and it
    cannot be passed by a slowly-drifting field. Returns the flux at each radius
    and the max/min ratio; 1.0 is a converged exterior.

    Measured on the proportional regime as shipped: 1719 -> 345 from r = 1.06R
    to 1.68R, a ratio of 5.0. The exterior is a transient, and the colony is
    consequently sitting in a near-full-strength reservoir.
    """
    g = sim.grid_resolution
    c = regime['L'] / 2.0
    yy, xx = np.mgrid[0:g, 0:g]
    rr = np.hypot(xx * sim.dx - c, yy * sim.dx - c)
    gy, gx = np.gradient(sim.nutrient_field, sim.dx)
    with np.errstate(invalid='ignore', divide='ignore'):
        dndr = (gx * (xx * sim.dx - c) + gy * (yy * sim.dx - c)) / np.maximum(rr, 1e-9)

    flux = []
    for f in radii_frac:
        rad = f * R
        if rad > 0.95 * c:                 # too close to the wall to be clean
            continue
        m = (rr > rad - 1.5 * sim.dx) & (rr < rad + 1.5 * sim.dx)
        if m.sum() < 8:
            continue
        flux.append(2.0 * np.pi * rad * regime['nutrient_D'] * float(np.mean(dndr[m])))
    if len(flux) < 2:
        return dict(flux=flux, ratio=np.nan, steady=False)
    lo, hi = min(flux), max(flux)
    ratio = hi / lo if lo > 0 else np.inf
    return dict(flux=flux, ratio=float(ratio), steady=bool(ratio < 1.15))


def measure_length_scale(sim, regime):
    """Fit the nutrient penetration depth l from the relaxed radial profile."""
    g = sim.grid_resolution
    c = regime['L'] / 2
    yy, xx = np.mgrid[0:g, 0:g]
    rr = np.hypot(xx * sim.dx - c, yy * sim.dx - c)
    bins = np.arange(0, regime['R0'] + 4, 2.5)
    cen, prof = [], []
    for i in range(len(bins) - 1):
        m = (rr >= bins[i]) & (rr < bins[i + 1])
        if m.any():
            cen.append(0.5 * (bins[i] + bins[i + 1]))
            prof.append(np.median(sim.nutrient_field[m]))
    cen, prof = np.array(cen), np.array(prof)
    depth = regime['R0'] - cen
    bc = regime['bc_value']
    band = (depth > 2.0) & (depth < 80.0) & (prof > 1e-3 * bc) & (prof < 0.9 * bc)
    if band.sum() < 4:
        return float('nan')
    slope = np.polyfit(depth[band], np.log(prof[band]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float('nan')


def active_fraction(sim, regime):
    """Fraction of cells above the quiescence threshold -- the rim/bulk ratio."""
    g = sim.grid_resolution
    n_active = 0
    for cell in sim.cells:
        i = int(cell.position[0] / sim.dx)
        j = int(cell.position[1] / sim.dx)
        if 0 <= j < g and 0 <= i < g:
            if sim.nutrient_field[j, i] > regime['qui_threshold']:
                n_active += 1
    return n_active / max(1, len(sim.cells))


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(regime, mode, seed, verbose=True):
    rng = np.random.default_rng(seed)
    sim = CellSimulation(make_config(regime, seed),
                         config_name=f"disp_{regime['name']}_k{mode}_s{seed}")
    center = np.array([regime['L'] / 2, regime['L'] / 2])

    sim.cells = seeded_colony(regime, mode, rng)
    set_phenotype(sim.cells, regime)
    eq_iters, eq_converged = equilibrate_nutrient(sim, regime)
    frac_active = active_fraction(sim, regime)
    l_measured = measure_length_scale(sim, regime)

    trace = []

    max_cfl = [0.0]

    def sample(step):
        pos = np.array([c.position for c in sim.cells])
        rad = np.array([c.radius for c in sim.cells])
        res = analyze_front(pos, rad, center)
        trace.append(dict(
            step=step, t=step * regime['dt'],
            n_cells=res['n_cells'], n_detached=res['n_detached'],
            mean_radius=res['mean_radius'], roughness=res['roughness'],
            a_rel=float(res['modes'][mode]),
            a_abs=float(res['modes'][mode] * res['mean_radius']),
        ))

    sample(0)
    for step in range(regime['steps']):
        set_phenotype(sim.cells, regime)     # daughters inherit the regime
        sim._simulation_step()
        # Advective CFL. The growth source can drive fast flow; above CFL ~ 1 the
        # semi-Lagrangian field advection is unstable and the run is meaningless,
        # so this is a hard validity condition, not a warning.
        cfl = float(np.max(np.abs(sim.fluid_velocity)) * sim.dt / sim.dx)
        max_cfl[0] = max(max_cfl[0], cfl)
        if (step + 1) % regime['sample_every'] == 0:
            sample(step + 1)
        if not sim.cells:
            break

    diag = dict(active_fraction=float(frac_active), l_measured=float(l_measured),
                eq_iters=int(eq_iters), eq_converged=bool(eq_converged),
                max_cfl=float(max_cfl[0]))
    return summarize(regime, mode, seed, trace, diag, verbose)


def summarize(regime, mode, seed, trace, diag, verbose):
    t = np.array([p['t'] for p in trace])
    a_rel = np.array([p['a_rel'] for p in trace])
    a_abs = np.array([p['a_abs'] for p in trace])
    R = np.array([p['mean_radius'] for p in trace])
    det = np.array([p['n_detached'] for p in trace])
    n = np.array([p['n_cells'] for p in trace])

    # Linear regime: up to where the relative amplitude has tripled (or the end).
    grew = np.flatnonzero(a_rel > 3.0 * a_rel[0])
    stop = grew[0] + 1 if grew.size else len(t)
    stop = max(stop, 4)                       # need enough points to fit

    fit_rel = fit_growth_rate(t[:stop], a_rel[:stop])
    fit_abs = fit_growth_rate(t[:stop], a_abs[:stop])

    advancing = bool(R[-1] > R[0])
    detach_frac = float(det.max() / max(1, n.max()))
    # Amplitude floor: a mode is unresolvable once its lobe is smaller than a
    # cell. If the fit window dips near it, the fitted rate is an underestimate.
    amp_floor = regime['cell_r'] / float(R[:stop].mean())
    above_floor = bool(a_rel[:stop].min() > 2.0 * amp_floor)
    ok = (advancing and detach_frac < 0.02 and diag['eq_converged']
          and above_floor and diag.get('max_cfl', 0.0) < 1.0)

    out = dict(
        regime=regime['name'], mode=int(mode), seed=int(seed),
        lambda_rel=fit_rel['lambda_'], stderr_rel=fit_rel['stderr'],
        r2_rel=fit_rel['r_squared'],
        lambda_abs=fit_abs['lambda_'], r2_abs=fit_abs['r_squared'],
        n_fit_points=fit_rel['n_points'],
        R_start=float(R[0]), R_end=float(R[-1]),
        dlnR_dt=float(np.log(R[-1] / R[0]) / (t[-1] - t[0])) if t[-1] > t[0] else 0.0,
        n_start=int(n[0]), n_end=int(n[-1]), cell_r=float(regime['cell_r']),
        max_detached_frac=detach_frac,
        amp_floor=float(amp_floor), above_floor=above_floor,
        advancing=advancing, valid=ok,
        a_rel_start=float(a_rel[0]), a_rel_end=float(a_rel[-1]),
        trace=trace, **diag,
    )
    if verbose:
        flag = ('OK ' if ok else
                'RECEDING' if not advancing else
                'SHEDDING' if detach_frac >= 0.02 else
                'NOT-EQUILIBRATED' if not diag['eq_converged'] else
                'CFL>1' if diag.get('max_cfl', 0.0) >= 1.0 else
                'BELOW-AMP-FLOOR')
        print(f"  k={mode:3d} seed={seed}: lam_rel={out['lambda_rel']:+.4f} "
              f"(r2={out['r2_rel']:.2f})  R:{R[0]:.1f}->{R[-1]:.1f}  "
              f"n:{n[0]}->{n[-1]}  det={detach_frac*100:.1f}%  "
              f"cfl={diag.get('max_cfl', 0.0):.2f}  [{flag}]",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(regime, modes, seeds):
    print(f"\n=== {regime['name']}: {len(modes)} modes x {len(seeds)} seeds ===",
          flush=True)
    t0 = time.time()
    results = []
    for mode in modes:
        for seed in seeds:
            results.append(run_one(regime, mode, seed))
    print(f"--- {len(results)} runs in {(time.time()-t0)/60:.1f} min", flush=True)

    out = os.path.join(HERE, f"dispersion_{regime['name']}.json")
    with open(out, 'w') as fh:
        json.dump(dict(regime={k: v for k, v in regime.items()},
                       results=results), fh, indent=1)
    print(f"Saved -> {out}")
    report(results)
    return results


def report(results):
    by_mode = {}
    for r in results:
        by_mode.setdefault(r['mode'], []).append(r)

    print(f"\n{'k':>4} {'lambda_rel':>18} {'r2':>6} {'valid':>6}  verdict")
    print('-' * 60)
    any_unstable = False
    for k in sorted(by_mode):
        rs = by_mode[k]
        lams = np.array([r['lambda_rel'] for r in rs], dtype=float)
        good = np.isfinite(lams)
        if not good.any():
            continue
        mean, std = lams[good].mean(), lams[good].std()
        r2 = np.mean([r['r2_rel'] for r in rs])
        nvalid = sum(r['valid'] for r in rs)
        unstable = mean - std > 0.0 and r2 > 0.8
        any_unstable |= unstable and nvalid == len(rs)
        print(f"{k:>4} {mean:>+10.4f} +/-{std:>6.4f} {r2:>6.2f} "
              f"{nvalid}/{len(rs):>4}  {'UNSTABLE' if unstable else 'stable'}")

    print('-' * 60)
    print("VERDICT:", "unstable band found" if any_unstable
          else "no unstable band -- front is stable in this regime")

    ks_sorted = sorted(by_mode)
    lam = [float(np.nanmean([r['lambda_rel'] for r in by_mode[k]]))
           for k in ks_sorted]
    k0 = marginal_mode(ks_sorted, lam)
    if k0 is not None:
        R = float(np.nanmean([r['R_start'] for r in results]))
        a = float(results[0].get('cell_r', 2.4)) if results else 2.4
        d0 = R / k0 ** 2
        kmax = ks_sorted[int(np.nanargmax(lam))]
        print(f"\nmarginal mode k0 = {k0:.2f}   (lambda crosses zero here)")
        print(f"implied capillary length d0 ~ R/k0^2 = {d0:.1f} units "
              f"= {d0/a:.1f} cell radii")
        print(f"most unstable mode measured: k = {kmax}"
              + ("  <-- at the edge of the swept range; the true peak may be "
                 "lower" if kmax == ks_sorted[0] else ""))


def marginal_mode(ks, lam):
    """Interpolate where lambda(k) crosses zero from positive to negative.

    The crossing is the marginal mode k0. For a growing front stabilized by
    surface tension the classic scaling is k0 ~ sqrt(R/d0), so k0 converts the
    measured dispersion relation into an estimate of the model's effective
    CAPILLARY LENGTH d0 -- the quantity that decides whether a fingering band
    can exist at all.
    """
    for i in range(len(ks) - 1):
        if lam[i] > 0.0 >= lam[i + 1]:
            f = lam[i] / (lam[i] - lam[i + 1])
            return ks[i] + f * (ks[i + 1] - ks[i])
    return None


def plot(regime_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = os.path.join(HERE, f"dispersion_{regime_name}.json")
    with open(path) as fh:
        data = json.load(fh)
    results = data['results']

    by_mode = {}
    for r in results:
        by_mode.setdefault(r['mode'], []).append(r)
    ks = sorted(by_mode)
    mean = [np.nanmean([r['lambda_rel'] for r in by_mode[k]]) for k in ks]
    std = [np.nanstd([r['lambda_rel'] for r in by_mode[k]]) for k in ks]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].errorbar(ks, mean, yerr=std, fmt='o-', capsize=3)
    ax[0].axhline(0.0, color='k', ls=':', alpha=0.7)
    ax[0].set(xlabel='angular mode k', ylabel=r'$\lambda_{rel}$  (1/time)',
              title=f"dispersion relation: {regime_name}")
    ax[0].grid(alpha=0.3)

    for k in ks:
        r = by_mode[k][0]
        t = [p['t'] for p in r['trace']]
        a = [p['a_rel'] for p in r['trace']]
        ax[1].semilogy(t, a, '-', alpha=0.7, label=f'k={k}')
    ax[1].set(xlabel='time', ylabel=r'relative amplitude $a_k$',
              title='mode amplitude (log): straight line = exponential')
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out = os.path.join(HERE, f"dispersion_{regime_name}.png")
    fig.savefig(out, dpi=110)
    print(f"Saved -> {out}")


def plot_compare(names, out_name='dispersion_compare'):
    """Overlay several dispersion relations, with the marginal mode marked."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in names:
        path = os.path.join(HERE, f'dispersion_{name}.json')
        if not os.path.exists(path):
            print(f"  (skipping {name}: not found)")
            continue
        with open(path) as fh:
            results = json.load(fh)['results']
        by = {}
        for r in results:
            by.setdefault(r['mode'], []).append(r)
        ks = sorted(by)
        mean = [float(np.nanmean([r['lambda_rel'] for r in by[k]])) for k in ks]
        err = [float(np.nanstd([r['lambda_rel'] for r in by[k]])) for k in ks]
        line = ax.errorbar(ks, mean, yerr=err, fmt='o-', capsize=3, label=name)
        k0 = marginal_mode(ks, mean)
        if k0 is not None:
            ax.axvline(k0, ls=':', alpha=0.5,
                       color=line.lines[0].get_color())

    ax.axhline(0.0, color='k', lw=1)
    ax.set(xlabel='angular mode k',
           ylabel=r'$\lambda_{rel}$  (1/time)',
           title='Colony-front dispersion relations\n'
                 '(dotted = marginal mode $k_0$; fingering needs an INTERIOR peak)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, f'{out_name}.png')
    fig.savefig(out, dpi=120)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'probe'

    if what == 'probe':
        named = {'prolif': PROLIFERATING, 'scale2': SCALE2,
                 'shortl': SHORT_L, 'wide': WIDE, 'chemo': CHEMO,
                 'prop': PROPORTIONAL, 'propbig': PROPORTIONAL_BIG,
                 'propfixed': PROPORTIONAL_FIXED}
        base = named.get(sys.argv[2] if len(sys.argv) > 2 else '', STAGE1)
        regime = dict(base, steps=120, sample_every=30)
        t0 = time.time()
        r = run_one(regime, mode=10, seed=1)
        a = regime['cell_r']
        print(f"\nprobe: {r['n_start']} cells | active rim "
              f"{r['active_fraction']*100:.0f}% | l={r['l_measured']:.1f} "
              f"(l/a={r['l_measured']/a:.1f}, R/l={regime['R0']/r['l_measured']:.1f}) "
              f"| equilibration {r['eq_iters']} iters converged={r['eq_converged']} "
              f"| R {r['R_start']:.1f}->{r['R_end']:.1f} "
              f"| {(time.time()-t0):.0f}s for {regime['steps']} steps")
    elif what == 'calibrate':
        sweep(CALIBRATION, modes=[4, 6, 8], seeds=[1, 2])
    elif what == 'stage1':
        sweep(STAGE1, modes=[3, 5, 7, 9, 11, 14, 17, 20], seeds=[1, 2, 3])
    elif what == 'stage3':
        sweep(STAGE3, modes=[3, 5, 7, 9, 11, 14, 17, 20], seeds=[1, 2, 3])
    elif what == 'prolif':
        sweep(PROLIFERATING, modes=[2, 3, 5, 7, 11], seeds=[1, 2, 3, 4])
    elif what == 'scale2':
        sweep(SCALE2, modes=[2, 3, 5, 7, 11], seeds=[1, 2, 3])
    elif what == 'shortl':
        sweep(SHORT_L, modes=[2, 3, 5, 7, 11], seeds=[1, 2, 3])
    elif what == 'wide':
        sweep(WIDE, modes=[3, 5, 7, 9], seeds=[1, 2])
    elif what == 'chemo':
        sweep(CHEMO, modes=[2, 3, 5, 7, 10, 14], seeds=[1, 2])
    elif what == 'prop':
        sweep(PROPORTIONAL, modes=[2, 3, 5, 7, 10, 14, 20], seeds=[1, 2, 3])
    elif what == 'propbig':
        sweep(PROPORTIONAL_BIG, modes=[3, 4, 6, 8, 10, 14], seeds=[1, 2])
    elif what == 'sigmascan':
        for reg in SIGMA_SCAN:
            sweep(reg, modes=[2, 5, 10, 20], seeds=[1])
    elif what == 'sigmahi':
        for reg in SIGMA_HIGH:
            sweep(reg, modes=[2, 3, 5, 7, 10, 14, 20], seeds=[1])
    elif what == 'stgrowth':
        for reg in GROWTHRATE_ST:
            sweep(reg, modes=[2, 3, 5, 7, 10], seeds=[1, 2])
    elif what == 'screening':
        for reg in SCREENING:
            sweep(reg, modes=[2, 3, 5, 7, 10, 14, 20], seeds=[1, 2])
    elif what == 'growthrate':
        for reg in GROWTHRATE:
            sweep(reg, modes=[2, 3, 5, 7, 10], seeds=[1, 2])
    elif what == 'propfixed':
        # Same regime, with the exterior nutrient field actually at steady state.
        # As shipped, the colony edge sat at 0.66 of the reservoir because the
        # initial guess put the whole exterior at full strength and the annulus
        # cannot relax in a feasible run; the quasi-steady seed puts it at 0.046,
        # matching the analytic 0.053. Every sweep above therefore ran ~14x too
        # nutrient-rich at the front -- the large-beta corner of the morphology
        # diagram, where the paper predicts lopsided colonies rather than fingers.
        sweep(PROPORTIONAL_FIXED, modes=[2, 3, 5, 7, 10, 14, 20], seeds=[1, 2, 3])
    elif what == 'peak':
        # Tie-break: is the k=3 maximum real, or is the top of the band flat?
        # lambda(3) - lambda(2) was +0.0003 with a k=3 scatter of +-0.0003, i.e.
        # 1 sigma. An interior maximum is the difference between "more unstable
        # modes" and "a SELECTED wavelength", so it is worth resolving properly.
        sweep(dict(CHEMO, name='chemotactic_peak'),
              modes=[2, 3, 4], seeds=[1, 2, 3, 4, 5, 6])
    elif what == 'plot':
        plot(sys.argv[2])
    elif what == 'compare':
        plot_compare(sys.argv[2:] or [
            'stage1_current_physics', 'ctrl_overlap_only',
            'ctrl_source_only', 'stage3_darcy_expansion'])
    else:
        raise SystemExit(f"unknown command {what!r}")


if __name__ == '__main__':
    main()
