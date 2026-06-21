"""The CellSimulation engine: orchestrates forces, hydrodynamics, transport,
biology, and the time-stepping loop."""
import os

import numpy as np

from .initializers import INITIALIZER_MAP
from .kernels.diffusion import (diffuse_field_numba, diffuse_field_implicit_numba,
                                advect_scalar_field_numba)
from .kernels.forces import (
    calculate_adhesion_forces_numba,
    calculate_repulsion_forces_numba,
    calculate_propulsion_forces_numba,
    resolve_overlaps_numba,
)
from .kernels.adhesion import calculate_differential_adhesion_forces_numba
from .kernels.mechanics import (
    velocity_gradient_numba,
    sample_gradient_at_cells_numba,
    strain_rate_and_axis,
)
from .kernels.shapes import contact_stress_celllist_numba
from .kernels.biology import cell_biology_step_numba
from .kernels.neighbors import (
    build_cell_list_numba,
    repulsion_forces_celllist_numba,
    contact_pressure_celllist_numba,
    adhesion_forces_celllist_numba,
    differential_adhesion_celllist_numba,
    resolve_overlaps_celllist_numba,
)
from .kernels.stokeslet import (
    update_fluid_velocity_numba,
    update_fluid_velocity_with_dipoles_numba,
    compute_cell_velocities_numba,
)
from .fluid.brinkman_fft import (
    solve_velocity, solve_velocity_variable_alpha, alpha_from_screening_length,
    solve_velocity_freeslip_box,
)
from .fluid.ibm import spread_forces_blob_numba, interpolate_velocity_blob_numba
from .kernels.fields import secrete_over_area_numba
from . import visualization
from . import io


class CellSimulation:
    def __init__(self, config, config_name='main'):
        self.config_name = config_name
        self.config = config

        # Reproducibility (issue #11). Seeding the global NumPy RNG here — before
        # cells/fields are built — makes all main-thread randomness (cell init,
        # division, type assignment, initializers) deterministic. The parallel
        # Numba kernels never draw their own randomness; the per-step random-walk
        # noise is generated here and passed in (see _calculate_forces).
        self.seed = config.get('seed')
        if self.seed is not None:
            np.random.seed(int(self.seed))

        self.physical_size = float(config['physical_size'])
        self.grid_resolution = int(config['grid_resolution'])
        self.dx = self.physical_size / self.grid_resolution

        self.dt = config['dt']
        self.nutrient_D = config['nutrient_D']
        self.chi_nutrient = config['chi_nutrient']

        # Diffusion solver: 'explicit' (forward-Euler, CFL-limited dt<dx^2/4D) or
        # 'implicit' (ADI/Peaceman-Rachford, unconditionally stable, exact steady
        # state -- use for stiff/high-D or quasi-steady studies). Both available.
        self.diffusion_solver = config.get('diffusion_solver', 'explicit')
        if self.diffusion_solver not in ('explicit', 'implicit'):
            raise ValueError("diffusion_solver must be 'explicit' or 'implicit'.")
        self._diffuse = (diffuse_field_implicit_numba
                         if self.diffusion_solver == 'implicit' else diffuse_field_numba)
        if self.diffusion_solver == 'implicit':
            print("INFO: Diffusion solver = implicit ADI (unconditionally stable).")

        # Nutrient uptake kinetics: Km (half-saturation) for Michaelis-Menten/Monod
        # saturating uptake. <= 0 -> first-order (linear) uptake (default).
        self.nutrient_uptake_saturation = float(config.get('nutrient_uptake_saturation', -1.0))

        # Multi-timescale operator splitting (PhysiCell-style decoupling of fast
        # transport from slow/expensive steps). Both default to 1 (no change).
        #  - diffusion_substeps M: diffuse the chemical fields M times per step
        #    with dt/M, so the cheap field relaxes toward quasi-steady and the
        #    explicit FTCS scheme stays stable at a large mechanics dt.
        #  - fluid_update_interval K: recompute the EXPENSIVE Brinkman FFT solve
        #    only every K steps (cell velocities are re-interpolated from the
        #    cached field every step). Bounded staleness error; ignored for the
        #    legacy Stokeslet path and forced to 1 when ECM is on (drag changes).
        self.diffusion_substeps = max(1, int(config.get('diffusion_substeps', 1)))
        self.fluid_update_interval = max(1, int(config.get('fluid_update_interval', 1)))
        self._step_count = 0
        if self.diffusion_substeps > 1:
            print(f"INFO: Diffusion sub-stepping ON ({self.diffusion_substeps} substeps/step).")
        if self.fluid_update_interval > 1:
            print(f"INFO: Fluid solve every {self.fluid_update_interval} steps "
                  f"(cached between).")

        # Nutrient boundary condition: 'dirichlet' holds edges at nutrient_bc_value
        # (external reservoir, use for branching/pattern studies).
        # 'neumann' (default) uses no-flux boundaries (isolated system).
        bc_type = config.get('nutrient_bc_type', 'neumann')
        if bc_type == 'dirichlet':
            self.nutrient_bc_value = float(config.get('nutrient_bc_value', 20.0))
            print(f"INFO: Nutrient BC = Dirichlet (held at {self.nutrient_bc_value})")
        else:
            self.nutrient_bc_value = -1.0  # signals Neumann to diffuse_field_numba
            print("INFO: Nutrient BC = Neumann (no-flux)")
        self.attractant_D = config['attractant_D']
        self.chi_attractant = config['chi_attractant']
        self.walk_speed = config['walk_speed']
        self.viscosity = config['viscosity']
        # Scalar adhesion strength (unused when an 'adhesion_matrix' is supplied).
        self.adhesion_strength = config.get('adhesion_strength', 0.0)

        # Linked-cell neighbor search for the pairwise force loops. Produces the
        # same forces as the brute-force O(N^2) kernels (to ~1e-10) but scales
        # near-linearly. On by default; set False to force brute force.
        self.use_neighbor_list = bool(config.get('use_neighbor_list', True))

        # Overlap-resolution sweeps per step. More sweeps relax dense/just-divided
        # packings faster, reducing transient overlaps after division (issue #21).
        self.overlap_iterations = int(config.get('overlap_iterations', 3))

        # Cell-shape mechanics (issue #22): each cell deforms into an
        # area-conserving ellipse under deviatoric contact stress, by a linear
        # viscoelastic law  d(eps)/dt = (chi*stress - eps)/tau, capped at a max
        # aspect ratio. Mechanics stay circular; shape is the elastic response.
        # Active<->passive (quiescence) transition: cells go passive where the
        # local nutrient drops below the threshold, freezing growth/division/
        # motility. Concentrates growth at the nutrient-exposed rim so the front
        # can finger (bacterial-colony branching needs this state switch).
        self.enable_quiescence = bool(config.get('enable_quiescence', False))
        self.quiescence_threshold = float(config.get('quiescence_nutrient_threshold', 5.0))

        # Mechanical feedback on proliferation (contact inhibition / homeostatic
        # pressure): cells whose compressive contact pressure exceeds the
        # threshold grow to full size but stop dividing. Off by default.
        self.enable_pressure_inhibition = bool(config.get('enable_pressure_inhibition', False))
        self.pressure_threshold = float(config.get('pressure_threshold', 1.0))
        if self.enable_pressure_inhibition:
            print(f"INFO: Pressure-inhibited growth ON "
                  f"(contact-pressure threshold = {self.pressure_threshold:.3g}).")

        # Gradient-directed division: daughters are placed up the local nutrient
        # gradient, so growth advances the front toward fresh nutrient (the
        # agent-level Mullins-Sekerka rule) rather than thickening it isotropically.
        self.directed_division = bool(config.get('directed_division', False))
        if self.directed_division:
            print("INFO: Gradient-directed division ON (daughters placed up-gradient).")
        if self.enable_quiescence:
            print(f"INFO: Active/passive quiescence ON "
                  f"(nutrient threshold = {self.quiescence_threshold:.3g}).")

        self.enable_cell_shape = bool(config.get('enable_cell_shape', False))
        self.shape_compliance = float(config.get('shape_compliance', 0.012))
        self.shape_relaxation_time = float(config.get('shape_relaxation_time', 0.6))
        self.shape_max_aspect = float(config.get('shape_max_aspect', 1.4))
        if self.enable_cell_shape:
            print(f"INFO: Cell-shape mechanics ON (compliance={self.shape_compliance:.3g}, "
                  f"relax_time={self.shape_relaxation_time:.3g}, "
                  f"max_aspect={self.shape_max_aspect:.3g}).")
        self.adhesion_cutoff_factor = config['adhesion_cutoff_factor']
        self.repulsion_strength = config['repulsion_strength']
        self.division_force_strength = config.get('division_force_strength', 10.0)

        self.hydrodynamics_model = config.get('hydrodynamics_model', 'monopole')
        print(f"INFO: Using '{self.hydrodynamics_model}' model for hydrodynamics.")

        # Mechanotransduction (issue #17): cells sense the local fluid strain
        # rate and align their polarity (nematically) toward the principal
        # strain axis at a rate proportional to shear. Optionally, polarity
        # drives an active propulsion (rheotaxis-like); off by default so the
        # sensing physics can be verified in isolation.
        self.enable_mechanotransduction = bool(config.get('enable_mechanotransduction', False))
        self.shear_alignment_rate = float(config.get('shear_alignment_rate', 1.0))
        self.polarity_propulsion_force = float(config.get('polarity_propulsion_force', 0.0))
        if self.enable_mechanotransduction:
            print(f"INFO: Mechanotransduction ON "
                  f"(alignment rate = {self.shear_alignment_rate:.3g}, "
                  f"polarity propulsion = {self.polarity_propulsion_force:.3g}).")

        # Fluid solver: 'stokeslet' (legacy free-space 2D Stokeslet sum, default)
        # or 'brinkman_fft' (FFT Brinkman solver + Immersed Boundary coupling).
        # The Brinkman path moves cells with the same velocity field that advects
        # the scalar fields and adds a substrate-drag screening length that
        # regularizes the 2D Stokes paradox.
        self.fluid_model = config.get('fluid_model', 'stokeslet')
        if self.fluid_model == 'brinkman_fft':
            # Screening length delta = sqrt(mu/alpha); default to a few grid cells'
            # worth of dish so interactions stay confined.
            self.brinkman_screening_length = float(
                config.get('brinkman_screening_length', 10.0 * self.dx)
            )
            self.brinkman_alpha = alpha_from_screening_length(
                self.viscosity, self.brinkman_screening_length
            )
            # IBM regularization width is tied to the cell radius (sigma =
            # ibm_reg_factor * radius), NOT the grid, so single-cell mobility is
            # grid-convergent (issue #16).
            self.ibm_reg_factor = float(config.get('ibm_reg_factor', 1.0))
            # Fluid boundary condition: 'periodic' (default) or 'freeslip_box'
            # (free-slip / no-penetration walls on all four sides, via a
            # mirror-symmetric extension of the periodic solver).
            self.fluid_boundary = config.get('fluid_boundary', 'periodic')
            if self.fluid_boundary not in ('periodic', 'freeslip_box'):
                raise ValueError(
                    "fluid_boundary must be 'periodic' or 'freeslip_box', "
                    f"got {self.fluid_boundary!r}")
            print(f"INFO: Fluid solver = Brinkman/FFT+IBM "
                  f"(screening length = {self.brinkman_screening_length:.3g}, "
                  f"ibm_reg_factor = {self.ibm_reg_factor:.3g}, "
                  f"boundary = '{self.fluid_boundary}').")
        else:
            print(f"INFO: Fluid solver = '{self.fluid_model}' (legacy Stokeslet).")

        # Optional spatial cutoff for Stokeslet calculations.
        # Set to a positive physical distance to skip far-field contributions and
        # speed up fluid velocity updates (useful for concentrated cell clusters).
        # A value of -1.0 (default) disables the cutoff for maximum accuracy.
        self.stokeslet_cutoff = float(config.get('stokeslet_cutoff', -1.0))

        # --- MODULAR INITIALIZATION ---
        setup_type = config.get('initial_setup_type', 'central_uniform')
        initializer_func = INITIALIZER_MAP.get(setup_type)
        if not initializer_func:
            raise ValueError(f"Unknown initial_setup_type: '{setup_type}'. "
                            f"Available options are: {list(INITIALIZER_MAP.keys())}")

        # Call the selected function to initialize cells and fields
        self.cells, self.nutrient_field = initializer_func(config, self.physical_size, self.grid_resolution)
        # Apply configured uptake kinetics to all initial cells (daughters inherit
        # via Cell.divide). Only overrides the default when a value is set.
        if self.nutrient_uptake_saturation > 0.0:
            for c in self.cells:
                c.uptake_saturation = self.nutrient_uptake_saturation
            print(f"INFO: Saturating (Michaelis-Menten) uptake ON "
                  f"(Km = {self.nutrient_uptake_saturation:.3g}).")
        # ----------------------------

        # --- Differential adhesion (optional) ---
        # If 'adhesion_matrix' is provided (KxK), adhesion strength is looked up
        # per cell-type pair and drives sorting/pattern formation; otherwise the
        # scalar 'adhesion_strength' is used for all pairs.
        self.adhesion_matrix = None
        if 'adhesion_matrix' in config:
            self.adhesion_matrix = np.asarray(config['adhesion_matrix'], dtype=np.float64)
            if self.adhesion_matrix.ndim != 2 or \
               self.adhesion_matrix.shape[0] != self.adhesion_matrix.shape[1]:
                raise ValueError("adhesion_matrix must be a square (K x K) array.")
            self._assign_cell_types(config)
            print(f"INFO: Differential adhesion ON "
                  f"({self.adhesion_matrix.shape[0]} cell types).")

        # Dynamic ECM (issue #18): cells deposit extracellular matrix into a
        # field that raises the local Brinkman drag (lowers permeability), so
        # flow reroutes around cell-built matrix. Requires fluid_model='brinkman_fft'.
        self.enable_ecm = bool(config.get('enable_ecm', False))
        self.ecm_secretion_rate = float(config.get('ecm_secretion_rate', 1.0))
        self.ecm_decay_rate = float(config.get('ecm_decay_rate', 0.01))
        self.ecm_drag_coeff = float(config.get('ecm_drag_coeff', 0.5))
        self.ecm_field = np.zeros((self.grid_resolution, self.grid_resolution))
        if self.enable_ecm:
            print(f"INFO: Dynamic ECM ON (secretion={self.ecm_secretion_rate:.3g}, "
                  f"decay={self.ecm_decay_rate:.3g}, drag_coeff={self.ecm_drag_coeff:.3g}).")

        self.attractant_field = np.zeros((self.grid_resolution, self.grid_resolution))
        self.fluid_velocity = np.zeros((self.grid_resolution, self.grid_resolution, 2))

        self.frames = []
        self._cfl_warned = False  # gate to print CFL warning only once
        self.output_dir = f"simulation_data_{self.config_name}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # --- Stability checks ---
        max_D = max(self.nutrient_D, self.attractant_D)
        diffusion_number = max_D * self.dt / (self.dx ** 2)
        if diffusion_number > 0.25:
            print(f"WARNING: Diffusion stability number = {diffusion_number:.4f} > 0.25. "
                  f"Explicit scheme may be unstable. Reduce dt or increase dx.")

    def _assign_cell_types(self, config):
        """Assign integer cell types for differential adhesion.

        config['cell_type_assignment']:
          'random' (default) — sample types using optional 'cell_type_fractions'.
          'left_right'        — type 0 on the left half of the dish, type 1 on
                                the right (useful for sorting at a wound).
        """
        k = self.adhesion_matrix.shape[0]
        assignment = config.get('cell_type_assignment', 'random')
        if assignment == 'left_right':
            mid = self.physical_size / 2.0
            for cell in self.cells:
                cell.cell_type = 0 if cell.position[0] < mid else 1
        else:
            fractions = config.get('cell_type_fractions')
            if fractions is not None:
                fractions = np.asarray(fractions, dtype=np.float64)
                fractions = fractions / fractions.sum()
            for cell in self.cells:
                cell.cell_type = int(np.random.choice(k, p=fractions))

    def _calculate_forces(self, cell_positions, radii):
        """Calculate all forces on cells, returned as (propulsion, monopolar) components.

        Propulsion forces are separated so the dipole hydrodynamics model can
        represent them as force dipoles (no net force on the fluid), while
        monopolar forces (adhesion, repulsion, division) act as point forces.
        """
        grad_nutrient_y, grad_nutrient_x = np.gradient(self.nutrient_field, self.dx)
        # Random-walk noise drawn here (seedable main thread) and injected into
        # the parallel kernel for reproducibility (issue #11).
        noise = np.random.standard_normal((len(self.cells), 2))
        propulsion_forces = calculate_propulsion_forces_numba(
            cell_positions, radii,
            grad_nutrient_x, grad_nutrient_y,
            self.chi_nutrient, self.walk_speed,
            self.config['max_propulsive_force'], self.dx,
            noise
        )

        if self.use_neighbor_list and len(radii) > 0:
            # Bin size must cover the largest interaction range: the adhesion
            # band (touch * cutoff_factor) for the largest pair, which also
            # covers the shorter repulsion (touch) range.
            bin_size = 2.0 * radii.max() * max(self.adhesion_cutoff_factor, 1.0)
            order, bin_start, nbx = build_cell_list_numba(
                cell_positions, self.physical_size, bin_size
            )
            if self.adhesion_matrix is not None:
                types = np.array([cell.cell_type for cell in self.cells], dtype=np.int64)
                adhesion_forces = differential_adhesion_celllist_numba(
                    cell_positions, radii, types, self.adhesion_matrix,
                    self.adhesion_cutoff_factor, order, bin_start, nbx, bin_size
                )
            else:
                adhesion_forces = adhesion_forces_celllist_numba(
                    cell_positions, radii, self.adhesion_strength,
                    self.adhesion_cutoff_factor, order, bin_start, nbx, bin_size
                )
            repulsion_forces = repulsion_forces_celllist_numba(
                cell_positions, radii, self.repulsion_strength,
                order, bin_start, nbx, bin_size
            )
        elif self.adhesion_matrix is not None:
            types = np.array([cell.cell_type for cell in self.cells], dtype=np.int64)
            adhesion_forces = calculate_differential_adhesion_forces_numba(
                cell_positions, radii, types, self.adhesion_matrix,
                self.adhesion_cutoff_factor
            )
            repulsion_forces = calculate_repulsion_forces_numba(
                cell_positions, radii, self.repulsion_strength
            )
        else:
            adhesion_forces = calculate_adhesion_forces_numba(
                cell_positions, radii, self.adhesion_strength, self.adhesion_cutoff_factor
            )
            repulsion_forces = calculate_repulsion_forces_numba(
                cell_positions, radii, self.repulsion_strength
            )

        division_forces = np.zeros_like(propulsion_forces)
        id_to_index = {cell.id: i for i, cell in enumerate(self.cells)}
        for i, cell in enumerate(self.cells):
            if cell.division_force_timer > 0 and cell.division_partner_id != -1:
                partner_idx = id_to_index.get(cell.division_partner_id)
                if partner_idx is not None:
                    partner_cell = self.cells[partner_idx]
                    delta = partner_cell.position - cell.position
                    distance = np.linalg.norm(delta)
                    if distance > 1e-6:
                        force_magnitude = self.division_force_strength / (distance**2 + 1)
                        direction = delta / distance
                        division_forces[i] -= force_magnitude * direction

        # Active propulsion along each cell's polarity (mechanotransduction
        # response, e.g. rheotaxis). Lagged: uses last step's polarity. Off when
        # polarity_propulsion_force == 0.
        if self.enable_mechanotransduction and self.polarity_propulsion_force > 0.0:
            for i, cell in enumerate(self.cells):
                propulsion_forces[i, 0] += self.polarity_propulsion_force * np.cos(cell.polarity)
                propulsion_forces[i, 1] += self.polarity_propulsion_force * np.sin(cell.polarity)

        # Quiescent (passive) cells do not self-propel (use last step's state).
        if self.enable_quiescence:
            for i, cell in enumerate(self.cells):
                if not cell.active:
                    propulsion_forces[i, 0] = 0.0
                    propulsion_forces[i, 1] = 0.0

        monopolar_forces = adhesion_forces + repulsion_forces + division_forces
        return propulsion_forces, monopolar_forces

    def _update_ecm(self):
        """Decay the ECM field and let each cell deposit matrix over its area.
        Returns the resulting spatially-varying Brinkman drag field."""
        if self.ecm_decay_rate > 0.0:
            self.ecm_field *= max(0.0, 1.0 - self.ecm_decay_rate * self.dt)
        amount = self.ecm_secretion_rate * self.dt
        for cell in self.cells:
            secrete_over_area_numba(cell.position, cell.radius, self.ecm_field,
                                    amount, self.dx)
        return self.brinkman_alpha + self.ecm_drag_coeff * self.ecm_field

    def _update_shapes(self, cell_positions, radii):
        """Evolve each cell's deviatoric strain toward chi * (deviatoric contact
        stress) by a viscoelastic relaxation, capped at shape_max_aspect.

            eps <- eps + (chi*stress - eps) * dt/tau
        """
        if len(radii) == 0:
            return
        bin_size = 2.0 * radii.max()      # repulsive contact range
        order, bin_start, nbx = build_cell_list_numba(
            cell_positions, self.physical_size, bin_size)
        sdev = contact_stress_celllist_numba(
            cell_positions, radii, self.repulsion_strength,
            order, bin_start, nbx, bin_size)
        rate = self.dt / self.shape_relaxation_time
        m_max = 0.5 * np.log(self.shape_max_aspect)
        chi = self.shape_compliance
        for i, cell in enumerate(self.cells):
            cell.exx += (chi * sdev[i, 0] - cell.exx) * rate
            cell.exy += (chi * sdev[i, 1] - cell.exy) * rate
            m = np.hypot(cell.exx, cell.exy)
            if m > m_max:
                f = m_max / m
                cell.exx *= f
                cell.exy *= f

    def _update_polarity(self, cell_positions, radii):
        """Align each cell's polarity nematically toward the local fluid strain
        axis at a rate proportional to the shear magnitude:

            dphi/dt = -k * shear_rate * sin(2 (phi - axis))

        with k = shear_alignment_rate. Fixed point at phi = axis (stable),
        phi = axis + pi/2 (unstable). Uses the just-solved fluid velocity.
        """
        grad = velocity_gradient_numba(self.fluid_velocity, self.dx)
        cell_grad = sample_gradient_at_cells_numba(cell_positions, radii, grad, self.dx)
        shear_rate, axis = strain_rate_and_axis(cell_grad)
        for i, cell in enumerate(self.cells):
            dphi = -self.shear_alignment_rate * shear_rate[i] * \
                np.sin(2.0 * (cell.polarity - axis[i])) * self.dt
            cell.polarity = (cell.polarity + dphi) % np.pi

    def _simulation_step(self):
        if not self.cells:
            return

        # Multi-timescale: recompute the expensive Brinkman solve only every
        # fluid_update_interval steps (always on step 0 and whenever ECM is on).
        recompute_fluid = (self._step_count % self.fluid_update_interval == 0
                           or self.enable_ecm)
        self._step_count += 1

        # 1. Build cell arrays
        cell_positions = np.array([cell.position for cell in self.cells])
        radii = np.array([cell.radius for cell in self.cells])

        # 2. Calculate forces (propulsion separated for dipole model)
        propulsion_forces, monopolar_forces = self._calculate_forces(cell_positions, radii)
        total_forces = propulsion_forces + monopolar_forces

        # 3. Compute fluid velocity on the grid (for scalar field advection).
        #    Stokes flow is instantaneous — velocity is fully determined by
        #    the current force distribution.
        #    For the Brinkman/IBM path the cell velocities come from the SAME
        #    solved field (computed here); the legacy Stokeslet path computes
        #    cell velocities separately in step 8.
        precomputed_cell_velocities = None
        if self.fluid_model == 'brinkman_fft':
            sigmas = self.ibm_reg_factor * radii   # physical regularization width
            if recompute_fluid:                    # multi-timescale: skip the
                force_density = spread_forces_blob_numba(   # expensive solve between
                    cell_positions, total_forces, sigmas,   # updates (cached field)
                    self.grid_resolution, self.grid_resolution, self.dx
                )
                if self.enable_ecm:
                    if self.fluid_boundary != 'periodic':
                        raise NotImplementedError(
                            "fluid_boundary='freeslip_box' is not yet supported "
                            "together with the variable-drag ECM solver; use "
                            "periodic boundaries with ECM.")
                    alpha_field = self._update_ecm()        # cell-remodeled drag field
                    self.fluid_velocity, self._ecm_iters, self._ecm_residual = \
                        solve_velocity_variable_alpha(
                            force_density, mu=self.viscosity, dx=self.dx,
                            alpha_field=alpha_field)
                elif self.fluid_boundary == 'freeslip_box':
                    self.fluid_velocity = solve_velocity_freeslip_box(
                        force_density, mu=self.viscosity, dx=self.dx,
                        alpha=self.brinkman_alpha,
                    )
                else:
                    self.fluid_velocity = solve_velocity(
                        force_density, mu=self.viscosity, dx=self.dx,
                        alpha=self.brinkman_alpha,
                    )
            precomputed_cell_velocities = interpolate_velocity_blob_numba(
                self.fluid_velocity, cell_positions, sigmas, self.dx
            )
        elif self.hydrodynamics_model == 'dipole':
            self.fluid_velocity[:] = 0.0
            orientations = np.zeros_like(propulsion_forces)
            norms = np.linalg.norm(propulsion_forces, axis=1)
            mask = norms > 1e-12
            orientations[mask] = propulsion_forces[mask] / norms[mask, np.newaxis]
            dipole_lengths = radii * 2.0
            update_fluid_velocity_with_dipoles_numba(
                self.fluid_velocity, cell_positions, monopolar_forces,
                propulsion_forces, orientations, dipole_lengths,
                self.viscosity, self.dx, self.stokeslet_cutoff
            )
        else:
            self.fluid_velocity[:] = 0.0
            update_fluid_velocity_numba(
                self.fluid_velocity, cell_positions, total_forces,
                self.viscosity, self.dx, self.stokeslet_cutoff
            )

        # 3b. Mechanotransduction: cells sense the fluid strain rate and align
        #     their polarity toward the principal strain axis (issue #17).
        if self.enable_mechanotransduction:
            self._update_polarity(cell_positions, radii)

        # 3c. Cell-shape mechanics: deform under contact stress (issue #22).
        if self.enable_cell_shape:
            self._update_shapes(cell_positions, radii)

        # 4. Check CFL stability for advection (warn once per simulation)
        if not self._cfl_warned:
            v_max = np.max(np.abs(self.fluid_velocity))
            cfl = v_max * self.dt / self.dx
            if cfl > 1.0:
                print(f"\nWARNING: CFL number = {cfl:.2f} > 1. "
                      f"Advection may be unstable. Consider reducing dt.")
                self._cfl_warned = True

        # 5. Advect scalar fields
        self.nutrient_field = advect_scalar_field_numba(
            self.nutrient_field, self.fluid_velocity, self.dt, self.dx
        )
        self.attractant_field = advect_scalar_field_numba(
            self.attractant_field, self.fluid_velocity, self.dt, self.dx
        )

        # 6. Diffuse scalar fields (solver selected by config: explicit/implicit;
        #    optionally sub-stepped M times with dt/M for stability/accuracy)
        dt_sub = self.dt / self.diffusion_substeps
        for _ in range(self.diffusion_substeps):
            self.nutrient_field = self._diffuse(
                self.nutrient_field, self.nutrient_D, dt_sub, self.dx,
                self.nutrient_bc_value
            )
            self.attractant_field = self._diffuse(
                self.attractant_field, self.attractant_D, dt_sub, self.dx
            )

        # 7. Cell biology (uptake, secretion, metabolism, growth, phase, death),
        #    batched into a single compiled kernel (replaces the per-cell loop).
        nutrient_to_read = np.copy(self.nutrient_field)
        nut_acc = np.array([c.nutrient_accumulated for c in self.cells])
        cons = np.array([c.consumption_rate for c in self.cells])
        secr = np.array([c.secretion_rate for c in self.cells])
        basal = np.array([c.basal_metabolism_rate for c in self.cells])
        active = np.array([c.active for c in self.cells])
        sat = np.array([c.uptake_saturation for c in self.cells])
        # Compressive contact pressure for mechanical feedback (only when needed)
        if self.enable_pressure_inhibition and len(radii) > 0:
            p_bin = 2.0 * radii.max() * max(self.adhesion_cutoff_factor, 1.0)
            p_order, p_start, p_nbx = build_cell_list_numba(
                cell_positions, self.physical_size, p_bin)
            pressure = contact_pressure_celllist_numba(
                cell_positions, radii, self.repulsion_strength,
                p_order, p_start, p_nbx, p_bin)
            for i, cell in enumerate(self.cells):
                cell.pressure = pressure[i]
        else:
            pressure = np.zeros(len(self.cells))
        c0 = self.cells[0]
        reached_div, alive = cell_biology_step_numba(
            cell_positions, radii, nut_acc, cons, secr, basal, active,
            self.nutrient_field, nutrient_to_read, self.attractant_field,
            self.dt, self.dx, c0.area_conserving, c0.min_radius, c0.max_radius,
            self.enable_quiescence, self.quiescence_threshold, sat,
            pressure, self.enable_pressure_inhibition, self.pressure_threshold)
        for i, cell in enumerate(self.cells):
            cell.nutrient_accumulated = nut_acc[i]
            cell.radius = radii[i]
            cell.active = bool(active[i])
            if reached_div[i] and cell.phase == 'GROWTH':
                cell.phase = 'DIVISION'
            if not alive[i]:
                cell.alive = False

        # 8. Compute cell velocities via mobility relation:
        #    v_k = F_k/(6*pi*mu*R_k) + sum_{j!=k} G(x_k,x_j).F_j
        #    Self-mobility uses Stokes drag; interactions use 2D Stokeslet.
        #    (Brinkman/IBM cell velocities were interpolated from the fluid in
        #    step 3 — reuse them so cells move with the field they advect.)
        if precomputed_cell_velocities is not None:
            cell_velocities = precomputed_cell_velocities
        else:
            cell_velocities = compute_cell_velocities_numba(
                cell_positions, total_forces, radii, self.viscosity, self.dx
            )
        for i, cell in enumerate(self.cells):
            cell.velocity = cell_velocities[i]
            cell.position += cell.velocity * self.dt

        # 9. Handle division and cleanup
        self._handle_division_and_death()
        self._resolve_overlaps()
        self._enforce_boundaries()

    def _handle_division_and_death(self):
        # Gradient-directed division: precompute the nutrient gradient so each
        # dividing cell can place its daughter up-gradient (toward fresh
        # nutrient), advancing the front instead of thickening it. This is the
        # agent-level "front advances along the flux" rule.
        grad_y = grad_x = None
        if self.directed_division:
            grad_y, grad_x = np.gradient(self.nutrient_field, self.dx)
        G = self.grid_resolution
        new_cells = []
        for cell in self.cells:
            if cell.active:                              # passive cells don't divide
                direction = None
                if self.directed_division:
                    i = int(cell.position[0] / self.dx)
                    j = int(cell.position[1] / self.dx)
                    if 0 <= j < G and 0 <= i < G:
                        direction = (grad_x[j, i], grad_y[j, i])   # up-gradient (x, y)
                new_cell = cell.divide(direction)
                if new_cell:
                    new_cells.append(new_cell)
            if cell.division_force_timer > 0:
                cell.division_force_timer -= 1
                if cell.division_force_timer == 0:
                    cell.division_partner_id = -1
        self.cells.extend(new_cells)
        self.cells = [c for c in self.cells if c.alive]

    def _resolve_overlaps(self):
        cell_positions = np.array([cell.position for cell in self.cells])
        radii = np.array([cell.radius for cell in self.cells])
        if self.use_neighbor_list and len(radii) > 0:
            # Parallel cell-list (Jacobi) sweeps: O(N) instead of O(N^2).
            bin_size = 2.0 * radii.max()      # overlap range = touching distance
            for _ in range(self.overlap_iterations):
                order, bin_start, nbx = build_cell_list_numba(
                    cell_positions, self.physical_size, bin_size)
                resolve_overlaps_celllist_numba(
                    cell_positions, radii, order, bin_start, nbx, bin_size)
        else:
            for _ in range(self.overlap_iterations):
                resolve_overlaps_numba(cell_positions, radii)
        for i, cell in enumerate(self.cells):
            cell.position = cell_positions[i].copy()

    def _enforce_boundaries(self):
        if not self.cells:
            return
        pos = np.array([c.position for c in self.cells])
        radii = np.array([c.radius for c in self.cells])
        np.clip(pos[:, 0], radii, self.physical_size - radii, out=pos[:, 0])
        np.clip(pos[:, 1], radii, self.physical_size - radii, out=pos[:, 1])
        for i, cell in enumerate(self.cells):
            cell.position = pos[i].copy()

    def run_simulation(self, steps, save_interval):
        print(f"Starting simulation '{self.config_name}' for {steps} steps.")
        for step in range(steps):
            self._simulation_step()

            if step > 0 and step % save_interval == 0:
                if self.config.get('enable_visualization', False):
                    self._save_frame(step)
                self._save_data_npz(step)

            print(f"\rStep {step+1}/{steps}, Cells: {len(self.cells)}", end="")
        print("\nSimulation finished.")

        if self.config.get('enable_visualization', False) and self.frames:
            print("Creating GIF...")
            self._create_gif()

    # --- Visualization / IO delegate to the dedicated modules ---
    def _save_frame(self, step):
        filepath = visualization.render_frame(
            self.nutrient_field, self.cells, self.physical_size, step, self.output_dir
        )
        self.frames.append(filepath)

    def _create_gif(self):
        visualization.create_gif(self.frames, self.config_name)

    def _save_data_npz(self, step):
        io.save_data_npz(
            self.cells, self.nutrient_field, self.attractant_field,
            self.fluid_velocity, self.config, self.output_dir,
            self.config_name, step
        )
