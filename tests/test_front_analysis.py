"""Tests for the colony-front diagnostics in cellflow.analysis.front.

These are analytic tests: synthetic colonies with a KNOWN front shape, so the
extracted mode amplitudes and growth rates can be checked against exact values
rather than against another simulation.

The final test class encodes the specific measurement bug that produced a
retracted result (see docs/giverso_replication.md): detached cells inflating the
apparent front roughness.
"""
import numpy as np
import pytest

from cellflow.analysis.front import (
    main_cluster_mask, front_radii, front_modes, roughness,
    fit_growth_rate, analyze_front,
)


def hex_disk(R0, cell_r=1.0, center=(0.0, 0.0), shape=None):
    """Hex-packed disk of touching cells; ``shape(theta) -> radius`` overrides R0."""
    spacing = 2.0 * cell_r
    pts = []
    n = int(2 * R0 / (spacing * np.sqrt(3) / 2)) + 3
    for j in range(-n, n + 1):
        y = j * spacing * np.sqrt(3) / 2
        xoff = (spacing / 2) if (j % 2) else 0.0
        for i in range(-n, n + 1):
            x = i * spacing + xoff
            r, th = np.hypot(x, y), np.arctan2(y, x)
            limit = shape(th) if shape is not None else R0
            if r <= limit:
                pts.append((center[0] + x, center[1] + y))
    pos = np.array(pts, dtype=np.float64)
    return pos, np.full(len(pos), cell_r)


def ring(R0, n_pts, center=(0.0, 0.0), cell_r=1.0, shape=None):
    """Cells placed exactly on a closed curve (front shape test fixture)."""
    th = np.linspace(-np.pi, np.pi, n_pts, endpoint=False)
    r = shape(th) if shape is not None else np.full(n_pts, R0)
    pos = np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th)], axis=1)
    return pos, np.full(n_pts, cell_r)


# --------------------------------------------------------------------------
# main_cluster_mask
# --------------------------------------------------------------------------
class TestMainClusterMask:

    def test_compact_disk_keeps_everything(self):
        pos, rad = hex_disk(20.0)
        mask = main_cluster_mask(pos, rad)
        assert mask.all()

    def test_detached_strays_are_excluded(self):
        pos, rad = hex_disk(20.0)
        strays = np.array([[60.0, 0.0], [0.0, 55.0], [-70.0, -70.0]])
        allpos = np.vstack([pos, strays])
        allrad = np.concatenate([rad, np.ones(len(strays))])
        mask = main_cluster_mask(allpos, allrad)
        assert mask[:len(pos)].all()
        assert not mask[len(pos):].any()
        assert (~mask).sum() == 3

    def test_larger_of_two_clusters_wins(self):
        big, rbig = hex_disk(15.0, center=(0.0, 0.0))
        small, rsmall = hex_disk(5.0, center=(200.0, 0.0))
        pos = np.vstack([big, small])
        rad = np.concatenate([rbig, rsmall])
        mask = main_cluster_mask(pos, rad)
        assert mask.sum() == len(big)
        assert mask[:len(big)].all()

    def test_touching_chain_is_one_cluster(self):
        # centres 2r apart => exactly touching => linked at any link_factor >= 1
        pos = np.array([[2.0 * i, 0.0] for i in range(25)])
        rad = np.ones(len(pos))
        assert main_cluster_mask(pos, rad, link_factor=1.05).all()

    def test_link_factor_threshold_is_respected(self):
        # two cells (r=1 each, sum=2) separated by 3.0 => ratio 1.5
        pos = np.array([[0.0, 0.0], [3.0, 0.0]])
        rad = np.ones(2)
        # link_factor 1.3 -> reach 2.6 < 3.0 -> separate (one is "the" cluster)
        assert main_cluster_mask(pos, rad, link_factor=1.3).sum() == 1
        # link_factor 1.6 -> reach 3.2 > 3.0 -> connected
        assert main_cluster_mask(pos, rad, link_factor=1.6).sum() == 2

    def test_negative_coordinates_handled(self):
        pos, rad = hex_disk(15.0, center=(-500.0, -300.0))
        assert main_cluster_mask(pos, rad).all()

    @pytest.mark.parametrize("n", [0, 1])
    def test_degenerate_inputs(self, n):
        pos = np.zeros((n, 2))
        rad = np.ones(n)
        assert main_cluster_mask(pos, rad).all()
        assert main_cluster_mask(pos, rad).shape == (n,)

    def test_scales_to_many_cells(self):
        pos, rad = hex_disk(60.0)          # ~2800 cells
        assert len(pos) > 2000
        assert main_cluster_mask(pos, rad).all()


# --------------------------------------------------------------------------
# front_radii
# --------------------------------------------------------------------------
class TestFrontRadii:

    def test_circle_recovered(self):
        pos, rad = ring(50.0, 2000)
        R = front_radii(pos, center=(0.0, 0.0))
        assert np.allclose(R, 50.0, atol=1e-9)

    def test_lobed_front_recovered(self):
        R0, eps, m = 100.0, 0.10, 6
        shape = lambda th: R0 * (1.0 + eps * np.cos(m * th))
        pos, rad = ring(R0, 8000, shape=shape)
        R = front_radii(pos, center=(0.0, 0.0), n_bins=360)
        theta = np.linspace(-np.pi, np.pi, 360, endpoint=False) + np.pi / 360.0
        # max-per-bin has a small positive bias; tolerance covers the bin width
        assert np.allclose(R, shape(theta), atol=0.02 * R0 * eps + 0.5)

    def test_offset_center_is_respected(self):
        c = (300.0, -120.0)
        pos, rad = ring(40.0, 1500, center=c)
        R = front_radii(pos, center=c)
        assert np.allclose(R, 40.0, atol=1e-9)

    def test_empty_bins_are_interpolated(self):
        # only 20 cells but 360 bins -> most bins empty
        pos, rad = ring(30.0, 20)
        R = front_radii(pos, center=(0.0, 0.0), n_bins=360)
        assert np.all(np.isfinite(R))
        assert np.allclose(R, 30.0, atol=1e-6)

    def test_takes_outermost_cell_per_bin(self):
        # a filled disk: the front must be the OUTER radius, not a mean
        pos, rad = hex_disk(40.0, cell_r=1.0)
        R = front_radii(pos, center=(0.0, 0.0), n_bins=120)
        assert R.mean() > 38.0

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            front_radii(np.zeros((0, 2)), center=(0.0, 0.0))


# --------------------------------------------------------------------------
# front_modes / roughness  (analytic normalization)
# --------------------------------------------------------------------------
class TestFrontModes:

    @pytest.mark.parametrize("m,eps", [(6, 0.10), (10, 0.05), (3, 0.20)])
    def test_single_mode_amplitude_is_exact(self, m, eps):
        """R = R0 (1 + eps cos(m theta))  =>  modes[m] == eps, others == 0."""
        theta = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        R = 100.0 * (1.0 + eps * np.cos(m * theta))
        a = front_modes(R)
        assert a[m] == pytest.approx(eps, rel=1e-9)
        others = np.delete(a[:40], m)
        assert np.max(np.abs(others)) < 1e-9

    def test_mean_mode_is_removed(self):
        theta = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        R = 50.0 * (1.0 + 0.1 * np.cos(4 * theta))
        assert front_modes(R)[0] == pytest.approx(0.0, abs=1e-12)

    def test_superposition_of_modes(self):
        theta = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        R = 80.0 * (1.0 + 0.06 * np.cos(5 * theta) + 0.03 * np.sin(11 * theta))
        a = front_modes(R)
        assert a[5] == pytest.approx(0.06, rel=1e-9)
        assert a[11] == pytest.approx(0.03, rel=1e-9)

    def test_amplitude_is_scale_invariant(self):
        """Mode amplitude is a FRACTION of mean radius -> independent of R0."""
        theta = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        a1 = front_modes(10.0 * (1 + 0.07 * np.cos(8 * theta)))
        a2 = front_modes(1000.0 * (1 + 0.07 * np.cos(8 * theta)))
        assert a1[8] == pytest.approx(a2[8], rel=1e-9)

    def test_roughness_of_circle_is_zero(self):
        assert roughness(np.full(360, 25.0)) == pytest.approx(0.0, abs=1e-12)

    def test_roughness_of_single_mode(self):
        """std/mean of R0(1+eps cos) is eps/sqrt(2)."""
        theta = np.linspace(-np.pi, np.pi, 3600, endpoint=False)
        eps = 0.12
        R = 100.0 * (1.0 + eps * np.cos(7 * theta))
        assert roughness(R) == pytest.approx(eps / np.sqrt(2.0), rel=1e-6)


# --------------------------------------------------------------------------
# fit_growth_rate
# --------------------------------------------------------------------------
class TestFitGrowthRate:

    def test_recovers_exact_exponential(self):
        t = np.linspace(0, 10, 25)
        lam = 0.37
        fit = fit_growth_rate(t, 0.01 * np.exp(lam * t))
        assert fit['lambda_'] == pytest.approx(lam, rel=1e-9)
        assert fit['r_squared'] == pytest.approx(1.0, abs=1e-12)

    def test_recovers_decay(self):
        t = np.linspace(0, 10, 25)
        fit = fit_growth_rate(t, 0.5 * np.exp(-0.21 * t))
        assert fit['lambda_'] == pytest.approx(-0.21, rel=1e-9)

    def test_noisy_transient_has_poor_fit_quality(self):
        """The retracted result's actual mode trace: ends up, but not exponential."""
        t = np.arange(7, dtype=float)
        a = np.array([0.066, 0.074, 0.051, 0.029, 0.025, 0.021, 0.032])
        fit = fit_growth_rate(t, a)
        assert fit['lambda_'] < 0.0            # net decay despite a mid-run bump
        assert fit['r_squared'] < 0.85         # and a bad exponential fit

    def test_too_few_points_returns_nan(self):
        fit = fit_growth_rate([0.0, 1.0], [0.1, 0.2])
        assert np.isnan(fit['lambda_'])
        assert fit['n_points'] == 2

    def test_non_positive_amplitudes_dropped(self):
        t = np.linspace(0, 5, 10)
        a = 0.02 * np.exp(0.5 * t)
        a[3] = 0.0
        a[7] = -1.0
        fit = fit_growth_rate(t, a)
        assert fit['n_points'] == 8
        assert fit['lambda_'] == pytest.approx(0.5, rel=1e-9)

    def test_stderr_is_zero_for_perfect_fit(self):
        t = np.linspace(0, 4, 12)
        fit = fit_growth_rate(t, np.exp(0.9 * t))
        assert fit['stderr'] == pytest.approx(0.0, abs=1e-9)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            fit_growth_rate([0, 1, 2], [1.0, 2.0])


# --------------------------------------------------------------------------
# The regression test: the bug that produced the retracted result
# --------------------------------------------------------------------------
class TestDetachedCellArtifact:
    """A SMOOTH colony that has shed cells must not read as a rough front."""

    @staticmethod
    def _colony_with_shed_rim(n_strays, seed=0):
        rng = np.random.default_rng(seed)
        pos, rad = hex_disk(60.0, cell_r=1.2)
        th = rng.uniform(-np.pi, np.pi, n_strays)
        rr = rng.uniform(75.0, 95.0, n_strays)      # clearly detached
        strays = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
        return (np.vstack([pos, strays]),
                np.concatenate([rad, np.full(n_strays, 1.2)]),
                len(pos))

    def test_strays_inflate_naive_roughness(self):
        pos, rad, n_body = self._colony_with_shed_rim(200)
        naive = roughness(front_radii(pos, center=(0.0, 0.0)))
        clean = roughness(front_radii(pos[:n_body], center=(0.0, 0.0)))
        # ~2.8x here; the 'clean' floor is hex-lattice faceting, not real roughness
        assert naive > 2.5 * clean

    def test_filtered_analysis_rejects_the_artifact(self):
        pos, rad, n_body = self._colony_with_shed_rim(200)
        res = analyze_front(pos, rad, center=(0.0, 0.0))
        clean = roughness(front_radii(pos[:n_body], center=(0.0, 0.0)))
        assert res['n_detached'] == 200
        assert res['roughness'] == pytest.approx(clean, rel=1e-9)

    def test_more_strays_do_not_change_the_filtered_front(self):
        """Roughness must be insensitive to how much rim has been shed."""
        vals = []
        for n in (0, 50, 200, 500):
            pos, rad, _ = self._colony_with_shed_rim(n)
            vals.append(analyze_front(pos, rad, center=(0.0, 0.0))['roughness'])
        assert max(vals) - min(vals) < 1e-9

    def test_seeded_mode_is_not_faked_by_strays(self):
        """Detached cells must not create a spurious mode-10 amplitude."""
        pos, rad, n_body = self._colony_with_shed_rim(300, seed=3)
        res = analyze_front(pos, rad, center=(0.0, 0.0))
        body = front_modes(front_radii(pos[:n_body], center=(0.0, 0.0)))
        assert res['modes'][10] == pytest.approx(body[10], rel=1e-9)

    def test_real_lobed_front_still_measured_correctly(self):
        """Filtering must not destroy a REAL seeded mode."""
        R0, eps, m = 60.0, 0.08, 8
        pos, rad = hex_disk(R0, cell_r=1.2,
                            shape=lambda th: R0 * (1 + eps * np.cos(m * th)))
        res = analyze_front(pos, rad, center=(0.0, 0.0))
        assert res['n_detached'] == 0
        assert res['modes'][m] == pytest.approx(eps, rel=0.25)


class TestEndToEndGrowthRateRecovery:
    """Positive control for the whole measurement chain.

    The calibration simulation only shows that the harness reports lambda < 0 on
    a stable front -- it cannot show the chain would DETECT growth if growth were
    there. Here we build synthetic colonies whose seeded mode grows at a known
    rate and check that analyze_front -> fit_growth_rate recovers it.
    """

    @staticmethod
    def _series(lam, mode=8, eps0=0.08, R0=120.0, n_t=8, dt=1.0, expand=0.0,
                filled=False, cell_r=1.0):
        times, amps = [], []
        for i in range(n_t):
            t = i * dt
            eps = eps0 * np.exp(lam * t)
            R = R0 * np.exp(expand * t)
            shape = lambda th, R=R, eps=eps: R * (1 + eps * np.cos(mode * th))
            if filled:
                pos, rad = hex_disk(R, cell_r=cell_r, shape=shape)
            else:
                pos, rad = ring(R, 4000, cell_r=cell_r, shape=shape)
            res = analyze_front(pos, rad, center=(0.0, 0.0))
            times.append(t)
            amps.append(res['modes'][mode])
        return np.array(times), np.array(amps)

    @pytest.mark.parametrize("lam", [0.25, 0.12, -0.18])
    def test_known_growth_rate_is_recovered(self, lam):
        t, a = self._series(lam)
        fit = fit_growth_rate(t, a)
        assert fit['lambda_'] == pytest.approx(lam, abs=0.02)
        assert fit['r_squared'] > 0.98

    @pytest.mark.parametrize("lam", [0.25, -0.18])
    def test_recovered_on_a_filled_hex_packed_colony(self, lam):
        """Same, on a realistically packed colony rather than a bare curve."""
        t, a = self._series(lam, filled=True)
        fit = fit_growth_rate(t, a)
        assert fit['lambda_'] == pytest.approx(lam, abs=0.05)

    def test_discretization_imposes_an_amplitude_floor(self):
        """A mode cannot be tracked below the cell-scale resolution ~ a/R.

        This is a real limit on any dispersion measurement in a discrete model:
        once the lobe amplitude drops under about one cell radius, the measured
        decay flattens and UNDERSTATES the true rate. Runs must keep the seeded
        amplitude above this floor for the whole fit window.
        """
        # amplitude decays below a cell radius: eps*R goes 1.8 -> 0.5 with a=1.2
        t, a = self._series(-0.18, eps0=0.03, R0=60.0, cell_r=1.2, filled=True)
        fit = fit_growth_rate(t, a)
        assert fit['lambda_'] > -0.16          # flattened, not the true -0.18
        # ... whereas well above the floor it is recovered
        t2, a2 = self._series(-0.18, eps0=0.08, R0=120.0, cell_r=1.0, filled=True)
        assert fit_growth_rate(t2, a2)['lambda_'] == pytest.approx(-0.18, abs=0.05)

    def test_sign_is_correct_on_a_growing_colony(self):
        """An amplifying mode on an EXPANDING colony must still read positive."""
        t, a = self._series(0.20, expand=0.05)
        fit = fit_growth_rate(t, a)
        assert fit['lambda_'] > 0.15

    def test_neutral_mode_reads_near_zero(self):
        t, a = self._series(0.0)
        fit = fit_growth_rate(t, a)
        assert abs(fit['lambda_']) < 0.02

    def test_detached_cells_do_not_create_false_growth(self):
        """Add more and more strays over time to a NEUTRAL front; the filtered
        chain must not report growth (this is exactly the retracted artifact)."""
        rng = np.random.default_rng(11)
        times, amps = [], []
        for i in range(8):
            pos, rad = hex_disk(60.0, cell_r=1.2,
                                shape=lambda th: 60.0 * (1 + 0.03 * np.cos(8 * th)))
            n_stray = 40 * i
            if n_stray:
                th = rng.uniform(-np.pi, np.pi, n_stray)
                rr = rng.uniform(75.0, 100.0, n_stray)
                strays = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
                pos = np.vstack([pos, strays])
                rad = np.concatenate([rad, np.full(n_stray, 1.2)])
            res = analyze_front(pos, rad, center=(0.0, 0.0))
            times.append(float(i))
            amps.append(res['modes'][8])
        fit = fit_growth_rate(np.array(times), np.array(amps))
        assert abs(fit['lambda_']) < 0.02
