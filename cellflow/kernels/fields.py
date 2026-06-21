"""Per-cell field sampling kernels: nutrient uptake, secretion, and averaging."""
import numpy as np
from numba import njit


@njit(cache=True)
def absorb_nutrient_numba(position, radius, nutrient_to_modify, nutrient_to_read,
                          dt, consumption_rate, dx, saturation=-1.0):
    """Nutrient uptake over the cell's area.

    Kinetics:
      - ``saturation <= 0`` (default): first-order (linear) uptake,
            rate(C) = consumption_rate * C            -- the original law.
      - ``saturation > 0``: Michaelis-Menten / Monod saturating uptake with
        half-saturation constant Km = ``saturation``,
            rate(C) = consumption_rate * C * Km/(Km + C).
        Here ``consumption_rate`` is the LOW-concentration rate constant, so the
        scheme reduces exactly to the linear law as Km -> inf, while the uptake
        per point saturates at consumption_rate*Km as C grows (real cells cannot
        consume arbitrarily fast). consumption_rate is spread over the cell area.
    """
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
                    C = nutrient_to_read[y, x]
                    rate = rate_per_point
                    if saturation > 0.0:
                        rate = rate_per_point * saturation / (saturation + C)
                    uptake = rate * C * dt
                    if uptake > C: uptake = C
                    nutrient_to_modify[y, x] -= uptake
                    total_uptake += uptake
    return total_uptake


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def _sample_scalar_field_numba(position, radius, field, dx):
    """Averages a 2D scalar field over the circular area of a cell.
    Helper used inside parallel batch functions."""
    x_center_idx = int(position[0] / dx)
    y_center_idx = int(position[1] / dx)
    r_idx = int(np.ceil(radius / dx))

    total_value = 0.0
    num_points = 0

    for i in range(-r_idx, r_idx + 1):
        for j in range(-r_idx, r_idx + 1):
            dist_sq = (i * dx)**2 + (j * dx)**2
            if dist_sq <= radius**2:
                y, x = y_center_idx + i, x_center_idx + j
                if 0 <= y < field.shape[0] and 0 <= x < field.shape[1]:
                    total_value += field[y, x]
                    num_points += 1

    if num_points > 0:
        return total_value / num_points
    if 0 <= y_center_idx < field.shape[0] and 0 <= x_center_idx < field.shape[1]:
        return field[y_center_idx, x_center_idx]
    return 0.0
