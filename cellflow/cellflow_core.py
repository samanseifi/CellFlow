import numpy as np
import os
import time
from numba import njit, prange
import matplotlib.pyplot as plt
import imageio

# --- Numba-optimized Helper Functions ---

@njit(parallel=True)
def diffuse_field_numba(field, D, dt, dx):
    """Generic function to diffuse any 2D field with no-flux boundaries."""
    new_field = np.copy(field)
    diffusion_factor = D * dt / (dx**2)
    for i in prange(1, field.shape[0] - 1):
        for j in prange(1, field.shape[1] - 1):
            laplacian = (field[i-1, j] + field[i+1, j] +
                         field[i, j-1] + field[i, j+1] - 4 * field[i, j])
            new_field[i, j] += diffusion_factor * laplacian
            if new_field[i, j] < 0:
                new_field[i, j] = 0
    return new_field

@njit
def absorb_nutrient_numba(position, radius, nutrient_to_modify, nutrient_to_read, dt, consumption_rate, dx):
    """Calculates nutrient uptake over the entire area of the cell."""
    total_uptake = 0.0
    x_center_idx, y_center_idx = int(position[0] / dx), int(position[1] / dx)
    r_idx = int(np.ceil(radius / dx))
    
    num_points_in_cell = np.pi * (radius/dx)**2
    if num_points_in_cell < 1: num_points_in_cell = 1
    rate_per_point = consumption_rate / num_points_in_cell

    for i in range(-r_idx, r_idx + 1):
        for j in range(-r_idx, r_idx + 1):
            dist_sq = (i*dx)**2 + (j*dx)**2
            if dist_sq <= radius**2:
                y, x = y_center_idx + i, x_center_idx + j
                if 0 <= y < nutrient_to_read.shape[0] and 0 <= x < nutrient_to_read.shape[1]:
                    uptake = rate_per_point * nutrient_to_read[y, x] * dt
                    if uptake > nutrient_to_read[y, x]: uptake = nutrient_to_read[y, x]
                    nutrient_to_modify[y, x] -= uptake
                    total_uptake += uptake
    return total_uptake

@njit(parallel=True)
def advect_scalar_field_numba(field, velocity, dt, dx):
    """Advects a scalar field using backward tracing and bilinear interpolation."""
    ny, nx = field.shape
    new_f = np.empty_like(field)
    
    # Pre-calculate dt/dx to avoid repeated division in the loop
    dt_div_dx = dt / dx

    for j in prange(ny):
        for i in prange(nx):
            # Back-trace position in grid-index coordinates
            x = i - velocity[j, i, 0] * dt_div_dx
            y = j - velocity[j, i, 1] * dt_div_dx

            # Clamp to domain boundaries for interpolation
            x = max(0.0, min(nx - 1.000001, x))
            y = max(0.0, min(ny - 1.000001, y))
            
            # Integer and fractional parts
            i0 = int(x)
            j0 = int(y)
            i1 = i0 + 1
            j1 = j0 + 1
            
            # Interpolation weights
            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            # Bilinear interpolation
            val = (s0 * (t0 * field[j0, i0] + t1 * field[j1, i0]) +
                   s1 * (t0 * field[j0, i1] + t1 * field[j1, i1]))
            
            new_f[j, i] = val
            
    return new_f


@njit
def secrete_over_area_numba(position, radius, attractant_field, total_secretion_amount, dx):
    """Distributes secreted attractant over the entire area of the cell."""
    x_center_idx, y_center_idx = int(position[0] / dx), int(position[1] / dx)
    r_idx = int(np.ceil(radius / dx))
    
    num_points_in_cell = np.pi * (radius/dx)**2
    if num_points_in_cell < 1: num_points_in_cell = 1
    secretion_per_point = total_secretion_amount / num_points_in_cell
    
    for i in range(-r_idx, r_idx + 1):
        for j in range(-r_idx, r_idx + 1):
            dist_sq = (i*dx)**2 + (j*dx)**2
            if dist_sq <= radius**2:
                y, x = y_center_idx + i, x_center_idx + j
                if 0 <= y < attractant_field.shape[0] and 0 <= x < attractant_field.shape[1]:
                    attractant_field[y, x] += secretion_per_point

@njit
def sample_field_at_cell_numba(position, radius, field, dx):
    """Averages a field (scalar or vector) over the area of a cell."""
    x_center_idx, y_center_idx = int(position[0] / dx), int(position[1] / dx)
    r_idx = int(np.ceil(radius / dx))
    
    is_vector_field = field.ndim == 3
    total_value = np.zeros(field.shape[2]) if is_vector_field else 0.0
    num_points = 0

    for i in range(-r_idx, r_idx + 1):
        for j in range(-r_idx, r_idx + 1):
            dist_sq = (i*dx)**2 + (j*dx)**2
            if dist_sq <= radius**2:
                y, x = y_center_idx + i, x_center_idx + j
                if 0 <= y < field.shape[0] and 0 <= x < field.shape[1]:
                    value = field[y, x, :] if is_vector_field else field[y, x]
                    total_value += value
                    num_points += 1
    
    if num_points > 0:
        return total_value / num_points
    else: 
        if 0 <= y_center_idx < field.shape[0] and 0 <= x_center_idx < field.shape[1]:
            return field[y_center_idx, x_center_idx, :] if is_vector_field else field[y_center_idx, x_center_idx]
        return np.zeros(field.shape[2]) if is_vector_field else 0.0

@njit(parallel=True)
def update_fluid_velocity_numba(fluid_velocity, positions, forces, viscosity, dx):
    """Calculates fluid velocity using the regularized Stokeslet (monopole) model."""
    h, w, _ = fluid_velocity.shape
    n = len(positions)
    eps = 2.0 * dx      # Regularization parameter
    eps2 = eps * eps
    pref = 1.0 / (8.0 * np.pi * viscosity)

    for gy in prange(h):
        for gx in range(w):
            xg = gx * dx
            yg = gy * dx
            vx_total, vy_total = 0.0, 0.0

            for k in range(n):
                px, py = positions[k, 0], positions[k, 1]
                fx, fy = forces[k, 0], forces[k, 1]

                rx = xg - px
                ry = yg - py
                r2 = rx*rx + ry*ry

                # Regularized denominators
                r_reg_inv = 1.0 / np.sqrt(r2 + eps2)
                r_reg_inv3 = r_reg_inv * r_reg_inv * r_reg_inv

                f_dot_r = fx * rx + fy * ry

                # Stokeslet formula
                vx = f_dot_r * rx * r_reg_inv3 + fx * r_reg_inv
                vy = f_dot_r * ry * r_reg_inv3 + fy * r_reg_inv

                vx_total += vx
                vy_total += vy

            fluid_velocity[gy, gx, 0] = pref * vx_total
            fluid_velocity[gy, gx, 1] = pref * vy_total

    return fluid_velocity

@njit(parallel=True)
def update_fluid_velocity_with_dipoles_numba(fluid_velocity, positions, monopolar_forces, propulsive_forces, orientations, cell_dipole_lengths, viscosity, dx):
    """
    Calculates fluid velocity using regularized Stokeslets for monopolar forces
    and regularized Stokes dipoles for propulsive (swimming) forces.
    """
    h, w, _ = fluid_velocity.shape
    num_cells = len(positions)
    eps = 2.0 * dx  # Regularization parameter
    eps2 = eps * eps
    pref = 1.0 / (8.0 * np.pi * viscosity)

    for gy in prange(h):
        for gx in range(w):
            xg = gx * dx
            yg = gy * dx
            vx_total, vy_total = 0.0, 0.0

            # Loop over each cell to sum its contribution
            for k in range(num_cells):
                px_k, py_k = positions[k, 0], positions[k, 1]

                # --- Part 1: Monopolar forces (adhesion, repulsion, etc.) ---
                f_mono_x, f_mono_y = monopolar_forces[k, 0], monopolar_forces[k, 1]
                if f_mono_x != 0.0 or f_mono_y != 0.0:
                    rx = xg - px_k
                    ry = yg - py_k
                    r2 = rx*rx + ry*ry
                    r_reg_inv = 1.0 / np.sqrt(r2 + eps2)
                    r_reg_inv3 = r_reg_inv * r_reg_inv * r_reg_inv
                    f_dot_r = f_mono_x * rx + f_mono_y * ry
                    
                    vx_total += f_dot_r * rx * r_reg_inv3 + f_mono_x * r_reg_inv
                    vy_total += f_dot_r * ry * r_reg_inv3 + f_mono_y * r_reg_inv

                # --- Part 2: Propulsive dipole (swimming force) ---
                f_prop_mag = np.sqrt(propulsive_forces[k,0]**2 + propulsive_forces[k,1]**2)
                if f_prop_mag > 1e-9:
                    # Unit vector for the swimming direction
                    ux, uy = orientations[k, 0], orientations[k, 1]
                    
                    # Effective length of the dipole (e.g., cell radius)
                    L = cell_dipole_lengths[k]
                    
                    # Position of the front (+F) and back (-F) poles of the dipole
                    pos_front = np.array([px_k + 0.5 * L * ux, py_k + 0.5 * L * uy])
                    pos_back = np.array([px_k - 0.5 * L * ux, py_k - 0.5 * L * uy])

                    # Contribution from the front pole (+F)
                    rx_f = xg - pos_front[0]
                    ry_f = yg - pos_front[1]
                    r2_f = rx_f*rx_f + ry_f*ry_f
                    r_reg_inv_f = 1.0 / np.sqrt(r2_f + eps2)
                    r_reg_inv3_f = r_reg_inv_f * r_reg_inv_f * r_reg_inv_f
                    f_dot_r_f = (f_prop_mag * ux) * rx_f + (f_prop_mag * uy) * ry_f
                    
                    vx_total += f_dot_r_f * rx_f * r_reg_inv3_f + (f_prop_mag * ux) * r_reg_inv_f
                    vy_total += f_dot_r_f * ry_f * r_reg_inv3_f + (f_prop_mag * uy) * r_reg_inv_f

                    # Contribution from the back pole (-F)
                    rx_b = xg - pos_back[0]
                    ry_b = yg - pos_back[1]
                    r2_b = rx_b*rx_b + ry_b*ry_b
                    r_reg_inv_b = 1.0 / np.sqrt(r2_b + eps2)
                    r_reg_inv3_b = r_reg_inv_b * r_reg_inv_b * r_reg_inv_b
                    f_dot_r_b = (-f_prop_mag * ux) * rx_b + (-f_prop_mag * uy) * ry_b

                    vx_total += f_dot_r_b * rx_b * r_reg_inv3_b + (-f_prop_mag * ux) * r_reg_inv_b
                    vy_total += f_dot_r_b * ry_b * r_reg_inv3_b + (-f_prop_mag * uy) * r_reg_inv_b

            fluid_velocity[gy, gx, 0] = pref * vx_total
            fluid_velocity[gy, gx, 1] = pref * vy_total

    return fluid_velocity

@njit(parallel=True)
def smoothly_damp_velocity_inside_cells_numba(velocity, positions, radii, dx):
    """
    Smoothly dampens the velocity to zero inside cells using a tanh function,
    creating a smoother transition than a hard cutoff.
    """
    ny, nx, _ = velocity.shape
    for idx in prange(len(positions)):
        cx, cy = positions[idx]
        radius = radii[idx]

        # The width of the transition region, e.g., 2 grid cells wide.
        # A smaller value gives a sharper (but still smooth) transition.
        transition_width = 2.0 * dx 

        # Define a bounding box to avoid checking the whole grid
        r_idx = int(np.ceil(radius / dx)) + 4
        xmin = max(0, int(cx / dx) - r_idx)
        xmax = min(nx, int(cx / dx) + r_idx + 1)
        ymin = max(0, int(cy / dx) - r_idx)
        ymax = min(ny, int(cy / dx) + r_idx + 1)

        for j in range(ymin, ymax):
            for i in range(xmin, xmax):
                # Distance from the current grid point to the cell center
                dist_from_center = np.sqrt(((i * dx) - cx)**2 + ((j * dx) - cy)**2)

                # tanh argument: shifts the center of the transition to the cell's radius
                arg = (dist_from_center - radius) / transition_width

                # The tanh function goes from -1 to 1. We scale it to go from 0 to 1.
                smoothing_factor = 0.5 * (1.0 + np.tanh(arg))

                # Apply the smoothing factor
                velocity[j, i, 0] *= smoothing_factor
                velocity[j, i, 1] *= smoothing_factor

@njit(parallel=True)
def zero_velocity_inside_cells_numba(velocity, positions, radii, dx):
    ny, nx, _ = velocity.shape
    for idx in prange(len(positions)):
        cx, cy = positions[idx]
        rad2    = (radii[idx] * 1.1)**2        # 10 % margin
        gx = int(cx / dx)
        gy = int(cy / dx)
        r_idx   = int(np.ceil(radii[idx] / dx)) + 2

        xmin = max(gx - r_idx, 0)
        xmax = min(gx + r_idx + 1, nx)
        ymin = max(gy - r_idx, 0)
        ymax = min(gy + r_idx + 1, ny)

        for j in range(ymin, ymax):
            dy = (j*dx - cy)
            for i in range(xmin, xmax):
                dx_ = (i*dx - cx)
                if dx_*dx_ + dy*dy <= rad2:
                    velocity[j, i, 0] = 0.0
                    velocity[j, i, 1] = 0.0

@njit(parallel=True)
def calculate_adhesion_forces_numba(positions, radii, adhesion_strength, adhesion_cutoff_factor):
    num_cells = len(positions)
    adhesion_forces = np.zeros((num_cells, 2))
    for i in prange(num_cells):
        for j in prange(i + 1, num_cells):
            delta = positions[j] - positions[i]
            distance = np.linalg.norm(delta)
            touching_dist = radii[i] + radii[j]
            cutoff_dist = touching_dist * adhesion_cutoff_factor
            if touching_dist < distance < cutoff_dist:
                force_magnitude = adhesion_strength * (distance - touching_dist)
                direction = delta / distance
                adhesion_forces[i] += force_magnitude * direction
                adhesion_forces[j] -= force_magnitude * direction
    return adhesion_forces

@njit(parallel=True)
def calculate_repulsion_forces_numba(positions, radii, repulsion_strength):
    num_cells = len(positions)
    repulsion_forces = np.zeros((num_cells, 2))
    for i in prange(num_cells):
        for j in prange(i + 1, num_cells):
            delta = positions[j] - positions[i]
            dist  = np.sqrt(delta[0]**2 + delta[1]**2)
            touch = radii[i] + radii[j]
            if 0.0 < dist < touch:
                overlap = touch - dist
                # --- new: exponentially stiff wall ----------------
                force_mag = repulsion_strength * np.exp(3.0 * overlap / touch)
                dir_vec   = delta / dist
                repulsion_forces[i] -= force_mag * dir_vec
                repulsion_forces[j] += force_mag * dir_vec
    return repulsion_forces


# --- Cell Definition ---
class Cell:
    next_id = 0
    def __init__(self, position, nutrient=20.0, just_divided_timer=0):
        self.id = Cell.next_id
        Cell.next_id += 1
        self.position = np.array(position, dtype=np.float64)
        self.nutrient_accumulated = nutrient
        self.consumption_rate = np.random.normal(0.2, 0.05) 
        self.secretion_rate = 1.0
        self.basal_metabolism_rate = 0.02
        self.alive = True
        self.phase = 'GROWTH'
        self.min_radius, self.max_radius = 2.0, 4.0
        self.radius = self.min_radius
        self.update_radius()
        self.velocity = np.array([0.0, 0.0])
        self.just_divided_timer = just_divided_timer
        self.division_partner_id = -1
        self.division_force_timer = 0

    def update_radius(self):
        growth_factor = (self.max_radius - self.min_radius) / 100.0 
        self.radius = self.min_radius + self.nutrient_accumulated * growth_factor
        self.radius = np.clip(self.radius, self.min_radius, self.max_radius)

    def update_phase(self):
        if self.radius >= self.max_radius and self.phase == 'GROWTH':
            self.phase = 'DIVISION'

    def absorb_nutrient(self, nutrient_to_modify, nutrient_to_read, dt, dx):
        return absorb_nutrient_numba(self.position, self.radius, nutrient_to_modify, nutrient_to_read, dt, self.consumption_rate, dx)

    def secrete_attractant(self, attractant_field, dt, dx):
        total_secretion = self.secretion_rate * dt
        secrete_over_area_numba(self.position, self.radius, attractant_field, total_secretion, dx)

    def check_death(self):
        if self.nutrient_accumulated < 0:
            self.alive = False

    def divide(self):
        if self.phase == 'DIVISION':
            self.nutrient_accumulated /= 2
            self.phase = 'GROWTH'
            self.update_radius()
            self.just_divided_timer = 5 
            
            daughter = Cell(self.position + np.random.randn(2) * self.radius, self.nutrient_accumulated, just_divided_timer=5)
            
            self.division_partner_id = daughter.id
            daughter.division_partner_id = self.id
            self.division_force_timer = 10
            daughter.division_force_timer = 10
            
            return daughter
        return None

# --- Modular Initializer Functions ---
def setup_central_uniform(config, physical_size, grid_resolution):
    """Initializes cells in a cluster at the center."""
    nutrient_field = np.ones((grid_resolution, grid_resolution)) * config.get('nutrient_bc_value', 20.0)
    center = physical_size / 2
    initial_cluster_radius = 20.0
    Cell.next_id = 0
    cells = [Cell(np.array([center, center]) + np.random.randn(2) * initial_cluster_radius) for _ in range(config['num_cells'])]
    return cells, nutrient_field

def setup_wound_healing(config, physical_size, grid_resolution):
    """Creates an initial state simulating a wound healing assay."""
    nutrient_field = np.ones((grid_resolution, grid_resolution)) * config.get('nutrient_bc_value', 100.0)
    cells = []
    Cell.next_id = 0
    
    wound_gap = config.get('wound_gap_width', 80.0)
    left_boundary = (physical_size / 2) - (wound_gap / 2)
    right_boundary = (physical_size / 2) + (wound_gap / 2)
    
    cell_spacing = config.get('initial_cell_spacing', 5.0)
    y_positions = np.arange(cell_spacing, physical_size, cell_spacing)
    
    # Left sheet
    x_positions_left = np.arange(cell_spacing, left_boundary, cell_spacing)
    for x in x_positions_left:
        for y in y_positions:
            cells.append(Cell(np.array([x, y])))
            
    # Right sheet
    x_positions_right = np.arange(right_boundary, physical_size, cell_spacing)
    for x in x_positions_right:
        for y in y_positions:
            cells.append(Cell(np.array([x, y])))
            
    return cells, nutrient_field

# --- NEW: MAP OF INITIALIZERS ---
# This dictionary maps a string name to the setup function.
INITIALIZER_MAP = {
    'central_uniform': setup_central_uniform,
    'wound_healing': setup_wound_healing,
}

class CellSimulation:
    def __init__(self, config, config_name='main'):
        # ... (initialization code remains the same) ...
        self.config_name = config_name
        self.config = config
        
        self.physical_size = float(config['physical_size'])
        self.grid_resolution = int(config['grid_resolution'])
        self.dx = self.physical_size / self.grid_resolution

        self.dt = config['dt']
        self.nutrient_D = config['nutrient_D']
        self.chi_nutrient = config['chi_nutrient']
        self.attractant_D = config['attractant_D']
        self.chi_attractant = config['chi_attractant']
        self.walk_speed = config['walk_speed']
        self.viscosity = config['viscosity']
        self.adhesion_strength = config['adhesion_strength']
        self.adhesion_cutoff_factor = config['adhesion_cutoff_factor']
        self.repulsion_strength = config['repulsion_strength']
        self.cell_mobility = config['cell_mobility']
        self.division_force_strength = config.get('division_force_strength', 10.0)
        
        self.hydrodynamics_model = config.get('hydrodynamics_model', 'monopole')
        print(f"INFO: Using '{self.hydrodynamics_model}' model for hydrodynamics.")
        
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
        self.output_dir = f"simulation_data_{self.config_name}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # --- NEW: Helper method to calculate all forces ---
    def _calculate_forces(self):
        cell_positions = np.array([cell.position for cell in self.cells])
        radii = np.array([cell.radius for cell in self.cells])
        
        # Propulsion (Constant Force Model)
        propulsion_forces = np.zeros((len(self.cells), 2))
        grad_nutrient_y, grad_nutrient_x = np.gradient(self.nutrient_field, self.dx)
        for i, cell in enumerate(self.cells):
            # The chemotaxis and random walk terms now define a desired DIRECTION, not a velocity.
            avg_grad_x = sample_field_at_cell_numba(cell.position, cell.radius, grad_nutrient_x, self.dx)
            avg_grad_y = sample_field_at_cell_numba(cell.position, cell.radius, grad_nutrient_y, self.dx)
            
            # This vector's direction is where the cell wants to go.
            force_direction_vec = self.chi_nutrient * np.array([avg_grad_x, avg_grad_y]) + \
                                  self.walk_speed * np.random.randn(2)
            
            # Normalize to get a pure direction (a unit vector)
            norm = np.linalg.norm(force_direction_vec)
            if norm > 0:
                unit_vec = force_direction_vec / norm
            else:
                unit_vec = np.zeros(2) # No force if no direction
            
            # Apply the cell's maximum propulsive force in the desired direction.
            propulsion_forces[i] = unit_vec * self.config['max_propulsive_force']

        # Inter-cell forces (these remain the same)
        adhesion_forces = calculate_adhesion_forces_numba(cell_positions, radii, self.adhesion_strength, self.adhesion_cutoff_factor)
        repulsion_forces = calculate_repulsion_forces_numba(cell_positions, radii, self.repulsion_strength)

        # Division forces (these remain the same)
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
        
        total_forces = propulsion_forces + adhesion_forces + repulsion_forces + division_forces
        return total_forces

    # --- NEW: A single, correctly ordered simulation step ---
    def _simulation_step(self):
        if not self.cells:
            return

        # 1. Calculate all forces acting on the cells
        cell_positions = np.array([cell.position for cell in self.cells])
        radii = np.array([cell.radius for cell in self.cells])
        total_forces_on_cells = self._calculate_forces()
        total_forces_on_fluid = self.cell_mobility * total_forces_on_cells

        # 2. Update fluid velocity using the new forces (with the corrected Stokeslet function)
        self.fluid_velocity = update_fluid_velocity_numba(self.fluid_velocity, cell_positions, total_forces_on_fluid, self.viscosity, self.dx)
        # zero_velocity_inside_cells_numba(self.fluid_velocity, cell_positions, radii, self.dx)
        # In _simulation_step method
        smoothly_damp_velocity_inside_cells_numba(self.fluid_velocity, cell_positions, radii, self.dx)

        # 3. Advect scalar fields using the new velocity field
        self.nutrient_field = advect_scalar_field_numba(self.nutrient_field, self.fluid_velocity, self.dt, self.dx)
        self.attractant_field = advect_scalar_field_numba(self.attractant_field, self.fluid_velocity, self.dt, self.dx)
        
        # 4. Diffuse scalar fields
        self.nutrient_field = diffuse_field_numba(self.nutrient_field, self.nutrient_D, self.dt, self.dx)
        self.attractant_field = diffuse_field_numba(self.attractant_field, self.attractant_D, self.dt, self.dx)

        # 5. Update cell biology (consumption, growth) and position
        nutrient_to_read = np.copy(self.nutrient_field)
        for i, cell in enumerate(self.cells):
            # Consumption/secretion happen AFTER advection/diffusion
            uptake = cell.absorb_nutrient(self.nutrient_field, nutrient_to_read, self.dt, self.dx)
            cell.nutrient_accumulated += uptake
            cell.secrete_attractant(self.attractant_field, self.dt, self.dx)
            cell.nutrient_accumulated -= cell.basal_metabolism_rate * self.dt

            # Update cell state
            cell.update_radius()
            cell.update_phase()
            cell.check_death()

            # Update cell position
            v_advection = sample_field_at_cell_numba(cell.position, cell.radius, self.fluid_velocity, self.dx)
            cell_drag_coeff = 6.0 * np.pi * self.viscosity * cell.radius
            v_active = total_forces_on_cells[i] / cell_drag_coeff
            
            # The v_advection already contains the effect from the cell's own force.
            # A simple way to model motility is to add the active part to the advected part.
            cell.velocity = self.cell_mobility * v_active + v_advection
            cell.position += cell.velocity * self.dt
        
        # 6. Handle division and cleanup
        self._handle_division_and_death()
        self._resolve_overlaps()
        self._enforce_boundaries()

    # --- NEW: Helper methods to clean up the main step function ---
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
        for a in range(len(self.cells)):
            for b in range(a + 1, len(self.cells)):
                ca, cb = self.cells[a], self.cells[b]
                delta = cb.position - ca.position
                dist = np.linalg.norm(delta)
                touch = ca.radius + cb.radius
                if dist < 1e-12:
                    shift = 0.5 * touch * np.random.randn(2)
                    ca.position -= shift
                    cb.position += shift
                elif dist < touch:
                    overlap = touch - dist
                    dir_vec = delta / dist
                    ca.position -= 0.5 * overlap * dir_vec
                    cb.position += 0.5 * overlap * dir_vec

    def _enforce_boundaries(self):
        for cell in self.cells:
            cell.position = np.clip(cell.position, cell.radius, self.physical_size - cell.radius)

    # --- MODIFIED: The run loop now calls the single step method ---
    def run_simulation(self, steps, save_interval):
        print(f"Starting simulation '{self.config_name}' for {steps} steps.")
        for step in range(steps):
            self._simulation_step()  # Call the new, correctly ordered step function
            
            if step > 0 and step % save_interval == 0:
                if self.config.get('enable_visualization', False):
                    self._save_frame(step)
                self._save_data_npz(step)
            
            print(f"\rStep {step+1}/{steps}, Cells: {len(self.cells)}", end="")
        print("\nSimulation finished.")

        if self.config.get('enable_visualization', False) and self.frames:
            print("Creating GIF...")
            self._create_gif()
            
    # --- Other methods (_save_frame, _create_gif, _save_data_npz) remain unchanged ---
        
    def _save_frame(self, step):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.nutrient_field, cmap='viridis', origin='lower', extent=[0, self.physical_size, 0, self.physical_size])
        
        for cell in self.cells:
            circle_outline = plt.Circle(cell.position, cell.radius, color='black', fill=False, lw=1)
            color = 'red' if cell.phase == 'DIVISION' else 'white'
            circle_body = plt.Circle(cell.position, cell.radius, color=color, alpha=0.7)
            ax.add_artist(circle_body)
            ax.add_artist(circle_outline)
            
        ax.set_title(f'Step: {step}, Cells: {len(self.cells)}')
        ax.set_xlim(0, self.physical_size)
        ax.set_ylim(0, self.physical_size)
        
        filepath = os.path.join(self.output_dir, f'frame_{step:04d}.png')
        plt.savefig(filepath)
        plt.close(fig)
        self.frames.append(filepath)

    def _create_gif(self):
        with imageio.get_writer(f"{self.config_name}_simulation.gif", mode='I', duration=0.1) as writer:
            for filename in self.frames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in self.frames:
            os.remove(filename)

    # --- MODIFIED: Method now saves data for a specific step ---
    def _save_data_npz(self, step):
        """Saves the simulation state at a specific step to a compressed NPZ file."""
        if not self.cells:
            return # Don't save if there are no cells

        cell_positions = np.array([cell.position for cell in self.cells])
        cell_velocities = np.array([cell.velocity for cell in self.cells])
        cell_radii = np.array([cell.radius for cell in self.cells])
        cell_ids = np.array([cell.id for cell in self.cells])
        cell_nutrients = np.array([cell.nutrient_accumulated for cell in self.cells])

        # --- MODIFIED: Filename now includes the step number ---
        filepath = os.path.join(self.output_dir, f'{self.config_name}_data_{step:04d}.npz')
        
        np.savez_compressed(
            filepath,
            step=step,
            cell_positions=cell_positions,
            cell_velocities=cell_velocities,
            cell_radii=cell_radii,
            cell_ids=cell_ids,
            cell_nutrients=cell_nutrients,
            final_nutrient_field=self.nutrient_field,
            final_attractant_field=self.attractant_field,
            final_fluid_velocity=self.fluid_velocity,
            config=self.config
        )
