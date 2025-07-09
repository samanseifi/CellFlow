import numpy as np
import matplotlib.pyplot as plt
import imageio
from numba import njit, prange
import time
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# --- Numba-optimized Helper Functions ---

@njit(parallel=True)
def diffuse_field_numba(field, D, dt):
    """Generic function to diffuse any 2D field with no-flux boundaries."""
    new_field = np.copy(field)
    for i in prange(1, field.shape[0] - 1):
        for j in prange(1, field.shape[1] - 1):
            laplacian = (field[i-1, j] + field[i+1, j] +
                         field[i, j-1] + field[i, j+1] - 4 * field[i, j])
            new_field[i, j] += D * laplacian * dt
            if new_field[i, j] < 0:
                new_field[i, j] = 0
    return new_field

@njit(parallel=True)
def diffuse_field_dirichlet_numba(field, D, dt, boundary_value):
    """Diffuses a field while holding the boundary at a constant concentration."""
    new_field = np.copy(field)
    height, width = field.shape
    for i in prange(1, height - 1):
        for j in prange(1, width - 1):
            laplacian = (field[i-1, j] + field[i+1, j] +
                         field[i, j-1] + field[i, j+1] - 4 * field[i, j])
            new_field[i, j] += D * laplacian * dt
            if new_field[i, j] < 0:
                new_field[i, j] = 0
    for i in prange(height):
        new_field[i, 0] = boundary_value
        new_field[i, width-1] = boundary_value
    for j in prange(width):
        new_field[0, j] = boundary_value
        new_field[height-1, j] = boundary_value
    return new_field

@njit
def absorb_nutrient_numba(position, radius, nutrient_to_modify, nutrient_to_read, dt, consumption_rate):
    """
    Calculates nutrient uptake over the entire area of the cell.
    Returns the total amount of nutrient absorbed.
    """
    total_uptake = 0.0
    x_center, y_center = int(position[0]), int(position[1])
    r_int = int(np.ceil(radius))
    
    num_points_in_cell = np.pi * radius**2
    if num_points_in_cell < 1:
        num_points_in_cell = 1
    rate_per_point = consumption_rate / num_points_in_cell

    for i in range(-r_int, r_int + 1):
        for j in range(-r_int, r_int + 1):
            if i**2 + j**2 <= radius**2:
                x, y = x_center + j, y_center + i
                
                if 0 <= x < nutrient_to_read.shape[1] and 0 <= y < nutrient_to_read.shape[0]:
                    uptake = rate_per_point * nutrient_to_read[y, x] * dt
                    if uptake > nutrient_to_read[y, x]:
                        uptake = nutrient_to_read[y, x]
                    
                    nutrient_to_modify[y, x] -= uptake
                    total_uptake += uptake
                    
    return total_uptake

@njit
def secrete_attractant_numba(position, attractant_field, secretion_rate, dt):
    x, y = int(position[0]), int(position[1])
    if 0 <= x < attractant_field.shape[1] and 0 <= y < attractant_field.shape[0]:
        attractant_field[y, x] += secretion_rate * dt

@njit(parallel=True)
def update_fluid_velocity_numba(fluid_velocity, positions, forces, viscosity):
    height, width, _ = fluid_velocity.shape
    num_cells = len(positions)
    stokeslet_k = 1.0 / (8.0 * np.pi * viscosity) if viscosity > 1e-6 else 0
    for i in prange(height):
        for j in prange(width):
            grid_pos = np.array([float(j), float(i)])
            total_vel = np.array([0.0, 0.0])
            if stokeslet_k > 0:
                for k in range(num_cells):
                    r_vec = grid_pos - positions[k]
                    r_dist = np.linalg.norm(r_vec)
                    if r_dist < 1e-6: continue
                    r_dist_inv = 1.0 / r_dist
                    r_dist_inv3 = r_dist_inv * r_dist_inv * r_dist_inv
                    f_dot_r = forces[k][0] * r_vec[0] + forces[k][1] * r_vec[1]
                    vel_contribution = stokeslet_k * ((forces[k] * r_dist_inv) + (f_dot_r * r_vec * r_dist_inv3))
                    total_vel += vel_contribution
            fluid_velocity[i, j, :] = total_vel
    return fluid_velocity

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
            distance = np.linalg.norm(delta)
            touching_dist = radii[i] + radii[j]
            if 0 < distance < touching_dist:
                overlap = touching_dist - distance
                force_magnitude = repulsion_strength * overlap
                direction = delta / distance
                repulsion_forces[i] -= force_magnitude * direction
                repulsion_forces[j] += force_magnitude * direction
    return repulsion_forces

# --- Cell Definition ---
class Cell:
    def __init__(self, position, nutrient=20.0, just_divided_timer=0):
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

    def update_radius(self):
        growth_factor = (self.max_radius - self.min_radius) / 100.0 
        self.radius = self.min_radius + self.nutrient_accumulated * growth_factor
        self.radius = np.clip(self.radius, self.min_radius, self.max_radius)

    def update_phase(self):
        if self.radius >= self.max_radius and self.phase == 'GROWTH':
            self.phase = 'DIVISION'

    def absorb_nutrient(self, nutrient_to_modify, nutrient_to_read, dt):
        return absorb_nutrient_numba(self.position, self.radius, nutrient_to_modify, nutrient_to_read, dt, self.consumption_rate)

    def check_death(self):
        if self.nutrient_accumulated < 0:
            self.alive = False

    def divide(self):
        if self.phase == 'DIVISION':
            self.nutrient_accumulated /= 2
            self.phase = 'GROWTH'
            self.update_radius()
            self.just_divided_timer = 5 
            return Cell(self.position + np.random.randn(2) * self.radius, self.nutrient_accumulated, just_divided_timer=5)
        return None

# --- Modular Initializer Functions ---
def setup_patchy(config):
    domain_size = np.array(config['domain_size'])
    nutrient_field = np.zeros(domain_size)
    x, y = np.meshgrid(np.arange(domain_size[0]), np.arange(domain_size[1]))
    for _ in range(10):
        center_x, center_y = np.random.rand(2) * domain_size
        richness = 40 * (0.5 + np.random.rand() * 0.5)
        size = 15 * (0.7 + np.random.rand() * 0.6)
        nutrient_field += richness * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * size**2))
    cells = [Cell(np.random.rand(2) * domain_size) for _ in range(config['num_cells'])]
    return cells, nutrient_field

def setup_central_uniform(config):
    domain_size = np.array(config['domain_size'])
    nutrient_field = np.ones(domain_size) * config.get('nutrient_bc_value', 20.0)
    center = domain_size / 2
    initial_radius = 20.0
    cells = [Cell(center + np.random.randn(2) * initial_radius) for _ in range(config['num_cells'])]
    return cells, nutrient_field

def setup_gradient(config):
    domain_size = np.array(config['domain_size'])
    x_coords = np.linspace(0, 50, domain_size[0])
    nutrient_field = np.tile(x_coords, (domain_size[1], 1))
    center = np.array([domain_size[0] * 0.1, domain_size[1] / 2])
    cells = [Cell(center + np.random.randn(2) * 10) for _ in range(config['num_cells'])]
    return cells, nutrient_field

def setup_scattered_uniform(config):
    domain_size = np.array(config['domain_size'])
    nutrient_field = np.ones(domain_size) * config.get('nutrient_bc_value', 20.0)
    cells = [Cell(np.random.rand(2) * domain_size) for _ in range(config['num_cells'])]
    return cells, nutrient_field

# --- Main Simulation Class ---
class CellSimulation:
    def __init__(self, config, config_name='main'):
        self.config_name = config_name
        self.config = config
        
        self.dt = config['dt']
        self.domain_size = np.array(config['domain_size'])
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
        self.boundary_condition = config.get('boundary_condition', 'no_flux')
        self.nutrient_bc_value = config.get('nutrient_bc_value', 0.0)
        
        SETUP_FUNCTIONS = {
            'patchy': setup_patchy,
            'central_uniform': setup_central_uniform,
            'gradient': setup_gradient,
            'scattered_uniform': setup_scattered_uniform,
        }
        setup_func = SETUP_FUNCTIONS.get(config['initial_setup'], setup_patchy)
        self.cells, self.nutrient_field = setup_func(config)

        self.attractant_field = np.zeros(self.domain_size)
        self.fluid_velocity = np.zeros((self.domain_size[1], self.domain_size[0], 2))

        self.frames = []
        self.output_dir = f"simulation_data_{self.config_name}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def update_fields(self):
        if self.boundary_condition == 'constant_concentration':
            self.nutrient_field = diffuse_field_dirichlet_numba(self.nutrient_field, self.nutrient_D, self.dt, self.nutrient_bc_value)
        else:
            self.nutrient_field = diffuse_field_numba(self.nutrient_field, self.nutrient_D, self.dt)
        self.attractant_field = diffuse_field_numba(self.attractant_field, self.attractant_D, self.dt)
        self.fluid_velocity *= 0.95

    def update_cells(self):
        enable_biology = self.config.get('enable_biology', True)
        constant_force = self.config.get('constant_force', None)
        
        propulsion_forces = np.zeros((len(self.cells), 2))
        if self.config.get('enable_propulsion', True):
            grad_nutrient_y, grad_nutrient_x = np.gradient(self.nutrient_field)
            grad_attractant_y, grad_attractant_x = np.gradient(self.attractant_field)
            for i, cell in enumerate(self.cells):
                if not cell.alive or cell.phase == 'DIVISION': continue
                pos_int = np.clip(cell.position.astype(int), 0, np.array(self.domain_size) - 1)
                grad_n = np.array([grad_nutrient_x[pos_int[1], pos_int[0]], grad_nutrient_y[pos_int[1], pos_int[0]]])
                grad_a = np.array([grad_attractant_x[pos_int[1], pos_int[0]], grad_attractant_y[pos_int[1], pos_int[0]]])
                v_chemotaxis = (self.chi_nutrient * grad_n) + (self.chi_attractant * grad_a)
                v_random = (np.random.randn(2) * self.walk_speed)
                propulsion_forces[i] = v_chemotaxis + v_random
        
        if constant_force is not None:
            if len(propulsion_forces) > 0: propulsion_forces[0] = constant_force

        cell_positions = np.array([cell.position for cell in self.cells])
        radii = np.array([cell.radius for cell in self.cells])
        
        adhesion_forces = calculate_adhesion_forces_numba(cell_positions, radii, self.adhesion_strength, self.adhesion_cutoff_factor)
        repulsion_forces = calculate_repulsion_forces_numba(cell_positions, radii, self.repulsion_strength)
        
        total_forces_on_fluid = propulsion_forces + adhesion_forces + repulsion_forces
        self.fluid_velocity = update_fluid_velocity_numba(self.fluid_velocity, cell_positions, total_forces_on_fluid, self.viscosity)
        
        nutrient_to_read = np.copy(self.nutrient_field)
        for i, cell in enumerate(self.cells):
            if not cell.alive: continue
            
            if enable_biology:
                uptake = cell.absorb_nutrient(self.nutrient_field, nutrient_to_read, self.dt)
                cell.nutrient_accumulated += uptake
                secrete_attractant_numba(cell.position, self.attractant_field, cell.secretion_rate, self.dt)
                cell.nutrient_accumulated -= cell.basal_metabolism_rate * self.dt
                cell.update_radius()
                cell.update_phase()
                cell.check_death()
            
            if not cell.alive: continue

            pos_int = np.clip(cell.position.astype(int), 0, np.array(self.domain_size) - 1)
            v_advection = self.fluid_velocity[pos_int[1], pos_int[0], :]
            
            net_force_on_cell = propulsion_forces[i] + adhesion_forces[i] + repulsion_forces[i]
            
            cell.velocity = self.cell_mobility * net_force_on_cell + v_advection
            cell.position += cell.velocity * self.dt

        new_cells = []
        for i, cell in enumerate(self.cells):
            cell.position = np.clip(cell.position, cell.radius, np.array(self.domain_size) - cell.radius)
            if enable_biology:
                new_cell = cell.divide()
                if new_cell: new_cells.append(new_cell)
            if cell.just_divided_timer > 0:
                cell.just_divided_timer -= 1

        self.cells.extend(new_cells)
        self.cells = [c for c in self.cells if c.alive]
        
    def run_simulation(self, steps=100, save_interval=50):
        enable_visualization = self.config.get('enable_visualization', True)

        if enable_visualization:
            cmap = cm.get_cmap('plasma')
            norm = plt.Normalize(vmin=2.0, vmax=4.0)

        for step in range(steps):
            self.update_fields()
            self.update_cells()
            
            if step > 0 and step % save_interval == 0:
                data_to_save = {
                    'fluid_velocity': self.fluid_velocity,
                    'nutrient_field': self.nutrient_field,
                    'attractant_field': self.attractant_field,
                    'cell_positions': np.array([cell.position for cell in self.cells]),
                    'cell_radii': np.array([cell.radius for cell in self.cells])
                }
                filename = os.path.join(self.output_dir, f"step_{step}.npz")
                np.savez_compressed(filename, **data_to_save)
                if not enable_visualization:
                    print(f"Saved data for step {step}")

            if enable_visualization:
                fig, ax = plt.subplots(figsize=(8,8))
                ax.set_facecolor('black')
                ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(self.nutrient_field, cmap='hot', origin='lower', vmin=0, vmax=20)
                
                colors = [(0, 1, 1, 0), (0, 1, 1, 1)]
                attractant_cmap = LinearSegmentedColormap.from_list('transparent_cyan', colors, N=100)
                ax.imshow(self.attractant_field, cmap=attractant_cmap, origin='lower', vmin=0, vmax=5)
                
                num_points = 15
                y_coords = np.linspace(0, self.domain_size[1] - 1, num_points)
                x_coords = np.linspace(0, self.domain_size[0] - 1, num_points)
                x, y = np.meshgrid(x_coords, y_coords)
                
                vx = self.fluid_velocity[y.astype(int), x.astype(int), 0]
                vy = self.fluid_velocity[y.astype(int), x.astype(int), 1]
                ax.streamplot(x, y, vx, vy, color='lightblue', linewidth=0.8, density=1.0, arrowstyle='->', arrowsize=1.0)

                for cell in self.cells:
                    # MODIFIED: Use facecolor instead of color to avoid warning
                    if cell.just_divided_timer > 0:
                        face_color = 'gray'
                    else:
                        face_color = 'gray'

                    circle = plt.Circle(cell.position, cell.radius, facecolor=face_color, fill=True, alpha=0.9,
                                        edgecolor='black', linewidth=0.5)
                    ax.add_artist(circle)
                
                ax.set_xlim(0, self.domain_size[0]); ax.set_ylim(0, self.domain_size[1])
                
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                self.frames.append(frame)
                plt.close(fig)
        
        if enable_visualization and self.frames:
            imageio.mimsave(f'{self.config_name}_simulation.gif', self.frames, fps=20)
