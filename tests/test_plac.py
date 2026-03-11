"""Tests for tibs.plac — thermodynamic, ODE, and Gillespie models."""

import numpy as np
import pytest

# Allow running from repo root or standalone
try:
    from tibs.plac import ThermodynamicPromoter, PLac, GillespiePLac
except ImportError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from tibs.plac import ThermodynamicPromoter, PLac, GillespiePLac


# ── ThermodynamicPromoter ─────────────────────────────────────────────────

class TestThermodynamicPromoter:

    def test_fold_change_monotonic_with_iptg(self):
        """More IPTG → higher fold-change (less repression)."""
        tp = ThermodynamicPromoter(R_total=10)
        iptg = np.logspace(-8, -2, 50)
        fc = tp.fold_change(iptg)
        assert np.all(np.diff(fc) >= 0), "fold-change must be non-decreasing with IPTG"

    def test_fold_change_saturates_near_one(self):
        """At very high IPTG, fold-change should approach 1 (fully de-repressed)."""
        tp = ThermodynamicPromoter(R_total=10)
        fc_high = tp.fold_change(1e-1)  # 100 mM — way above saturation
        assert fc_high > 0.95, f"expected ~1.0 at saturating IPTG, got {fc_high}"

    def test_stronger_repression_with_more_laci(self):
        """More LacI copies → lower fold-change at a fixed sub-saturating IPTG."""
        fc_low_R = ThermodynamicPromoter(R_total=5).fold_change(1e-6)
        fc_high_R = ThermodynamicPromoter(R_total=200).fold_change(1e-6)
        assert fc_low_R > fc_high_R

    def test_state_probabilities_sum_to_one(self):
        tp = ThermodynamicPromoter()
        probs = tp.state_probabilities(1e-5)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-10

    def test_dose_response_shape(self):
        tp = ThermodynamicPromoter()
        iptg, fc = tp.dose_response(n_points=100)
        assert iptg.shape == (100,)
        assert fc.shape == (100,)

    def test_no_repressor_means_no_regulation(self):
        """With R=0, fold-change is identically 1 at all IPTG."""
        tp = ThermodynamicPromoter(R_total=0)
        fc = tp.fold_change(np.logspace(-8, -2, 20))
        np.testing.assert_allclose(fc, 1.0, atol=1e-10)


# ── PLac ODE model ────────────────────────────────────────────────────────

class TestPLac:

    def test_simulate_returns_correct_shape(self):
        m = PLac(IPTG=1e-3)
        res = m.simulate(t_span=(0, 50), dt=0.1)
        n_steps = len(np.arange(0, 50, 0.1))
        assert res.x.shape == (n_steps, 2)

    def test_induced_reaches_steady_state(self):
        """Fully induced system should reach a non-zero steady state."""
        m = PLac(IPTG=1e-3)
        res = m.simulate(t_span=(0, 500), dt=0.1)
        final_protein = res.x[-1, 1]
        assert final_protein > 0.1, "protein should accumulate when induced"

    def test_repressed_stays_low(self):
        """Without IPTG, expression should be very low."""
        m = PLac(IPTG=0.0)
        res = m.simulate(t_span=(0, 500), dt=0.1)
        final_protein = res.x[-1, 1]
        # Some leaky expression is expected; just check it's much lower
        m_induced = PLac(IPTG=1e-3)
        res_induced = m_induced.simulate(t_span=(0, 500), dt=0.1)
        assert final_protein < 0.1 * res_induced.x[-1, 1]


# ── GillespiePLac ─────────────────────────────────────────────────────────

class TestGillespiePLac:

    def test_run_returns_valid_arrays(self):
        g = GillespiePLac(IPTG=1e-3, seed=42)
        t, X = g.run(t_end=50.0)
        assert t.ndim == 1
        assert X.ndim == 2
        assert X.shape[1] == 2
        assert len(t) == len(X)

    def test_molecule_counts_nonnegative(self):
        g = GillespiePLac(IPTG=1e-3, seed=0)
        _, X = g.run(t_end=100.0)
        assert np.all(X >= 0)

    def test_seed_reproducibility(self):
        g1 = GillespiePLac(IPTG=1e-4, seed=123)
        g2 = GillespiePLac(IPTG=1e-4, seed=123)
        t1, X1 = g1.run(t_end=50.0)
        t2, X2 = g2.run(t_end=50.0)
        np.testing.assert_array_equal(X1, X2)

    def test_ensemble_returns_correct_count(self):
        g = GillespiePLac(IPTG=1e-3, seed=0)
        ensemble = g.run_ensemble(n_cells=10, t_end=30.0)
        assert len(ensemble) == 10

    def test_mean_protein_increases_with_iptg(self):
        """Ensemble mean protein at high IPTG > low IPTG."""
        low = GillespiePLac(IPTG=1e-8, seed=0)
        high = GillespiePLac(IPTG=1e-3, seed=0)
        ens_low = low.run_ensemble(n_cells=50, t_end=200.0)
        ens_high = high.run_ensemble(n_cells=50, t_end=200.0)
        mean_low = np.mean([X[-1, 1] for _, X in ens_low])
        mean_high = np.mean([X[-1, 1] for _, X in ens_high])
        assert mean_high > mean_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
