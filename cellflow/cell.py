"""The Cell agent: state, growth, nutrient exchange, and division."""
import numpy as np

from .kernels.fields import absorb_nutrient_numba, secrete_over_area_numba


class Cell:
    next_id = 0

    def __init__(self, position, nutrient=20.0, just_divided_timer=0, cell_type=0):
        self.id = Cell.next_id
        Cell.next_id += 1
        self.position = np.array(position, dtype=np.float64)
        self.cell_type = cell_type
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
        # Polarity director angle (radians, nematic: defined mod pi). Aligns
        # toward the local fluid strain axis when mechanotransduction is on.
        # Initialized deterministically from the id (golden-angle spread) so it
        # gives varied orientations WITHOUT consuming the global RNG stream
        # (which would shift seeded results, breaking reproducibility #11).
        self.polarity = (self.id * 2.399963229728653) % np.pi
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

            daughter = Cell(self.position + np.random.randn(2) * self.radius,
                            self.nutrient_accumulated, just_divided_timer=5,
                            cell_type=self.cell_type)
            daughter.polarity = self.polarity     # inherit orientation

            self.division_partner_id = daughter.id
            daughter.division_partner_id = self.id
            self.division_force_timer = 10
            daughter.division_force_timer = 10

            return daughter
        return None
