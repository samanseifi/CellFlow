"""The Cell agent: state, growth, nutrient exchange, and division."""
import numpy as np

from .kernels.fields import absorb_nutrient_numba, secrete_over_area_numba


class Cell:
    next_id = 0

    def __init__(self, position, nutrient=20.0, just_divided_timer=0, cell_type=0,
                 area_conserving=False):
        self.id = Cell.next_id
        Cell.next_id += 1
        self.position = np.array(position, dtype=np.float64)
        self.cell_type = cell_type
        # Growth mode: 'area_conserving' grows AREA linearly with nutrient
        # (radius ~ sqrt) so division (halving nutrient) halves area -> total
        # area is conserved. The legacy default grows radius linearly with
        # nutrient (area grows quadratically; division is not area-conserving).
        self.area_conserving = area_conserving
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
        # Deviatoric shape strain (exx, exy); (0,0) = round. Evolves under
        # contact stress when cell-shape mechanics are enabled. Area-conserving.
        self.exx = 0.0
        self.exy = 0.0
        self.just_divided_timer = just_divided_timer
        self.division_partner_id = -1
        self.division_force_timer = 0

    def update_radius(self):
        if self.area_conserving:
            # Area grows linearly with nutrient (mass); radius from area. At the
            # division threshold nutrient=100 -> radius=max_radius; halving
            # nutrient -> area/2 -> radius = max_radius/sqrt(2). Two daughters
            # then total the mother's area exactly.
            frac = max(0.0, self.nutrient_accumulated / 100.0)
            area = np.pi * self.max_radius ** 2 * frac
            area_floor = np.pi * (0.5 * self.min_radius) ** 2   # stability floor
            self.radius = min(np.sqrt(max(area, area_floor) / np.pi), self.max_radius)
        else:
            growth_factor = (self.max_radius - self.min_radius) / 100.0
            self.radius = self.min_radius + self.nutrient_accumulated * growth_factor
            self.radius = np.clip(self.radius, self.min_radius, self.max_radius)

    def update_phase(self):
        if self.radius >= self.max_radius and self.phase == 'GROWTH':
            self.phase = 'DIVISION'

    def shape_axes(self):
        """Area-conserving ellipse (semi-major a, semi-minor b, angle in rad)
        from the deviatoric strain. exx=exy=0 -> a=b=radius (circle).
        a*b = radius^2 always (area conserved)."""
        m = np.sqrt(self.exx * self.exx + self.exy * self.exy)
        a = self.radius * np.exp(m)
        b = self.radius * np.exp(-m)
        angle = 0.5 * np.arctan2(self.exy, self.exx)
        return a, b, angle

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

            # Place the daughter just *touching* the parent (separation = sum of
            # radii) along a random direction, rather than a zero-mean Gaussian
            # offset that would bury it inside the parent (issue #21).
            d = np.random.randn(2)
            norm = np.sqrt(d[0] ** 2 + d[1] ** 2)
            direction = d / norm if norm > 1e-9 else np.array([1.0, 0.0])
            daughter = Cell(self.position.copy(), self.nutrient_accumulated,
                            just_divided_timer=5, cell_type=self.cell_type,
                            area_conserving=self.area_conserving)
            daughter.position = self.position + direction * (self.radius + daughter.radius)
            daughter.polarity = self.polarity     # inherit orientation

            self.division_partner_id = daughter.id
            daughter.division_partner_id = self.id
            self.division_force_timer = 10
            daughter.division_force_timer = 10

            return daughter
        return None
