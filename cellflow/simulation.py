"""The CellSimulation engine: orchestrates forces, hydrodynamics, transport,
biology, and the time-stepping loop."""
import os

import numpy as np

from .initializers import INITIALIZER_MAP
from .kernels.diffusion import diffuse_field_numba, advect_scalar_field_numba
from .kernels.forces import (
    calculate_adhesion_forces_numba,
    calculate_repulsion_forces_numba,
    calculate_propulsion_forces_numba,
    resolve_overlaps_numba,
)
from .kernels.stokeslet import (
    update_fluid_velocity_numba,
    update_fluid_velocity_with_dipoles_numba,
    compute_cell_velocities_numba,
)
from . import visualization
from . import io


class CellSimulation:
    def __init__(self, config, config_name='main'):
        self.config_name = config_name
        self.config = config

        self.physical_size = float(config['physical_size'])
        self.grid_resolution = int(config['grid_resolution'])
        self.dx = self.physical_size / self.grid_resolution

        self.dt = config['dt']
        self.nutrient_D = config['nutrient_D']
        self.chi_nutrient = config['chi_nutrient']

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
        self.adhesion_strength = config['adhesion_strength']
        self.adhesion_cutoff_factor = config['adhesion_cutoff_factor']
        self.repulsion_strength = config['repulsion_strength']
        self.division_force_strength = config.get('division_force_strength', 10.0)

        self.hydrodynamics_model = config.get('hydrodynamics_model', 'monopole')
        print(f"INFO: Using '{self.hydrodynamics_model}' model for hydrodynamics.")

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
        # ----------------------------

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

    def _calculate_forces(self, cell_positions, radii):
        """Calculate all forces on cells, returned as (propulsion, monopolar) components.

        Propulsion forces are separated so the dipole hydrodynamics model can
        represent them as force dipoles (no net force on the fluid), while
        monopolar forces (adhesion, repulsion, division) act as point forces.
        """
        grad_nutrient_y, grad_nutrient_x = np.gradient(self.nutrient_field, self.dx)
        propulsion_forces = calculate_propulsion_forces_numba(
            cell_positions, radii,
            grad_nutrient_x, grad_nutrient_y,
            self.chi_nutrient, self.walk_speed,
            self.config['max_propulsive_force'], self.dx
        )

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

        monopolar_forces = adhesion_forces + repulsion_forces + division_forces
        return propulsion_forces, monopolar_forces

    def _simulation_step(self):
        if not self.cells:
            return

        # 1. Build cell arrays
        cell_positions = np.array([cell.position for cell in self.cells])
        radii = np.array([cell.radius for cell in self.cells])

        # 2. Calculate forces (propulsion separated for dipole model)
        propulsion_forces, monopolar_forces = self._calculate_forces(cell_positions, radii)
        total_forces = propulsion_forces + monopolar_forces

        # 3. Compute fluid velocity on the grid (for scalar field advection).
        #    Stokes flow is instantaneous — velocity is fully determined by
        #    the current force distribution, so we zero the field first.
        self.fluid_velocity[:] = 0.0
        if self.hydrodynamics_model == 'dipole':
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
            update_fluid_velocity_numba(
                self.fluid_velocity, cell_positions, total_forces,
                self.viscosity, self.dx, self.stokeslet_cutoff
            )

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

        # 6. Diffuse scalar fields
        self.nutrient_field = diffuse_field_numba(
            self.nutrient_field, self.nutrient_D, self.dt, self.dx,
            self.nutrient_bc_value
        )
        self.attractant_field = diffuse_field_numba(
            self.attractant_field, self.attractant_D, self.dt, self.dx
        )

        # 7. Cell biology (consumption, growth)
        nutrient_to_read = np.copy(self.nutrient_field)
        for i, cell in enumerate(self.cells):
            uptake = cell.absorb_nutrient(
                self.nutrient_field, nutrient_to_read, self.dt, self.dx
            )
            cell.nutrient_accumulated += uptake
            cell.secrete_attractant(self.attractant_field, self.dt, self.dx)
            cell.nutrient_accumulated -= cell.basal_metabolism_rate * self.dt
            cell.update_radius()
            cell.update_phase()
            cell.check_death()

        # 8. Compute cell velocities via mobility relation:
        #    v_k = F_k/(6*pi*mu*R_k) + sum_{j!=k} G(x_k,x_j).F_j
        #    Self-mobility uses Stokes drag; interactions use 2D Stokeslet.
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
        new_cells = []
        for cell in self.cells:
            new_cell = cell.divide()
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
        resolve_overlaps_numba(cell_positions, radii)
        for i, cell in enumerate(self.cells):
            cell.position = cell_positions[i].copy()

    def _enforce_boundaries(self):
        for cell in self.cells:
            cell.position = np.clip(cell.position, cell.radius, self.physical_size - cell.radius)

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
