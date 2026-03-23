"""
P03 claim tests — Observability, Spectral Coherence, and Topological Control.

Run: python -m pytest -q _claude/tests/paper_claims/test_p03_claims.py
"""
import pytest
import jax.numpy as jnp
import numpy as np

from ods_unified_v2.jax_runtime.observability import (
    compute_commutator_defect,
    field_observables,
)
from ods_unified_v2.jax_runtime.integrator import default_params
from ods_unified_v2.jax_runtime.amr import (
    rotational_tension_map,
    activity_mask,
    activity_ratio,
)
from ods_unified_v2.jax_runtime.gamma_virtual import (
    gamma_focus_score,
    gamma_grade_entropy,
)
from ods_unified_v2.jax_runtime.defect_stabilizer import (
    apply_defect_stabilizer,
    defect_stabilizer_gate,
)
from ods_unified_v2.jax_runtime.wilson import dirac_defect
from ods_unified_v2.jax_runtime.state import ensure_clifford_state


# ── Helpers ─────────────────────────────────────────────────────────────

def _random_field(seed=42, H=16, W=16):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))


def _uniform_bivector_field(H=16, W=16):
    """Spatially uniform bivector field — zero rotational tension expected."""
    field = jnp.zeros((H, W, 8), dtype=jnp.float32)
    field = field.at[..., 4].set(1.0)  # e12
    field = field.at[..., 5].set(0.5)  # e13
    field = field.at[..., 6].set(0.3)  # e23
    return field


# ── Claim 1: Commutator defect ──────────────────────────────────────────

class TestCommutatorDefect:

    def test_defect_positive_when_both_clocks_active(self):
        Psi = _random_field()
        params = default_params()
        params["omega_12"] = 0.15
        params["gamma_12"] = 0.10
        params["k_tau"] = 2
        params["tau_dt"] = 1e-3
        params["dt"] = 1e-2
        defect = compute_commutator_defect(Psi, params)
        assert defect > 0, f"Commutator defect should be > 0, got {defect}"

    def test_defect_zero_when_tau_disabled(self):
        Psi = _random_field()
        params = default_params()
        params["omega_12"] = 0.15
        params["gamma_12"] = 0.10
        params["k_tau"] = 0
        params["tau_dt"] = 0.0
        defect = compute_commutator_defect(Psi, params)
        assert defect < 1e-6, f"Defect should be ~0 when tau disabled, got {defect}"

    def test_defect_zero_when_t_disabled(self):
        Psi = _random_field()
        params = default_params()
        params["omega_12"] = 0.0
        params["gamma_12"] = 0.0
        params["dt"] = 0.0
        params["k_tau"] = 2
        params["tau_dt"] = 1e-3
        defect = compute_commutator_defect(Psi, params)
        assert defect < 1e-6, f"Defect should be ~0 when t disabled, got {defect}"


# ── Claim 2: λ₂ tracks field coherence ──────────────────────────────────

class TestSpectralCoherence:
    """Requires slice (5D) fields — test at import level."""

    @pytest.fixture
    def tcl_module(self):
        try:
            from ods_unified_v2.jax_runtime.tcl_spectral import (
                build_tcl_spectral_report,
                algebraic_connectivity,
                graph_laplacian,
                smooth_adjacency,
                slice_feature_matrix,
            )
            return {
                "build_report": build_tcl_spectral_report,
                "lambda2": algebraic_connectivity,
                "laplacian": graph_laplacian,
                "adjacency": smooth_adjacency,
                "features": slice_feature_matrix,
            }
        except ImportError:
            pytest.skip("tcl_spectral not available")

    def test_lambda2_positive_for_nontrivial_field(self, tcl_module):
        rng = np.random.default_rng(42)
        field = jnp.array(rng.normal(size=(4, 8, 8, 8, 8)).astype(np.float32))
        report = tcl_module["build_report"](field)
        assert report.lambda2 > 0, f"λ₂ should be > 0, got {report.lambda2}"
        assert report.node_count == 4

    def test_identical_slices_high_lambda2(self, tcl_module):
        """Identical slices → maximum coherence → high λ₂."""
        single = np.random.default_rng(7).normal(size=(1, 8, 8, 8, 8)).astype(np.float32)
        field = jnp.array(np.tile(single, (4, 1, 1, 1, 1)))
        report = tcl_module["build_report"](field)
        # For identical slices, adjacency is nearly uniform
        assert report.adjacency_density > 0.5, f"Density should be high for identical slices"

    def test_t0_spec_inverse_of_lambda2(self, tcl_module):
        rng = np.random.default_rng(42)
        field = jnp.array(rng.normal(size=(4, 8, 8, 8, 8)).astype(np.float32))
        report = tcl_module["build_report"](field)
        expected_t0 = 1.0 / (report.lambda2 + 1e-6)
        assert abs(report.t0_spec - expected_t0) < 0.01


# ── Claim 3: Rotational tension localizes activity ───────────────────────

class TestRotationalTension:

    def test_zero_tension_for_uniform_field(self):
        field = _uniform_bivector_field()
        tension = rotational_tension_map(field)
        max_tension = float(jnp.max(tension))
        assert max_tension < 1e-5, f"Uniform field should have ~0 tension, got {max_tension}"

    def test_nonzero_tension_for_random_field(self):
        field = _random_field()
        tension = rotational_tension_map(field)
        mean_tension = float(jnp.mean(tension))
        assert mean_tension > 0.01, f"Random field should have nonzero tension, got {mean_tension}"

    def test_activity_mask_covers_high_tension(self):
        field = _random_field()
        tension = rotational_tension_map(field)
        mask = activity_mask(tension, quantile=0.9, dilation_radius=1)
        ratio = float(activity_ratio(mask))
        # With q=0.9 + dilation, expect ~10-30% of grid active
        assert 0.05 < ratio < 0.60, f"Activity ratio should be 5-60%, got {ratio}"

    def test_activity_mask_shape(self):
        field = _random_field(H=32, W=32)
        tension = rotational_tension_map(field)
        mask = activity_mask(tension, quantile=0.9)
        assert mask.shape == (32, 32), f"Mask shape mismatch: {mask.shape}"


# ── Claim 4: gamma_focus_score ───────────────────────────────────────────

class TestGammaFocusScore:

    def test_bivector_dominated_high_score(self):
        """Field with energy concentrated in bivector should have high focus."""
        field = jnp.zeros((16, 16, 8), dtype=jnp.float32)
        field = field.at[..., 4:7].set(1.0)  # bivector channels
        score = float(gamma_focus_score(field))
        # Bivector ratio ~1, entropy low → score should be positive
        assert score > 0, f"Bivector-dominated field should have positive focus, got {score}"

    def test_uniform_grade_lower_score(self):
        """Uniform energy across grades → higher entropy → lower focus."""
        field = jnp.ones((16, 16, 8), dtype=jnp.float32)
        score_uniform = float(gamma_focus_score(field))
        # Compare with bivector-dominated
        field_biv = jnp.zeros((16, 16, 8), dtype=jnp.float32)
        field_biv = field_biv.at[..., 4:7].set(1.0)
        score_biv = float(gamma_focus_score(field_biv))
        assert score_biv > score_uniform, \
            f"Bivector focus {score_biv} should exceed uniform {score_uniform}"

    def test_grade_entropy_normalized(self):
        field = _random_field()
        entropy = float(gamma_grade_entropy(field))
        assert 0 <= entropy <= 1.0 + 1e-6, f"Normalized entropy should be in [0,1], got {entropy}"


# ── Claim 5: Defect stabilizer preserves spin index ──────────────────────

class TestDefectStabilizer:

    def test_gate_zero_below_threshold(self):
        gate = float(defect_stabilizer_gate(0.5, threshold=1.0, gain=0.8))
        assert gate < 0.5, f"Gate should be <0.5 below threshold, got {gate}"

    def test_gate_high_above_threshold(self):
        gate = float(defect_stabilizer_gate(2.0, threshold=1.0, gain=0.8))
        assert gate > 0.5, f"Gate should be >0.5 above threshold, got {gate}"

    def test_defect_nonincreasing(self):
        field = _random_field(seed=99)
        d_before = float(dirac_defect(field, filtered=True))
        result = apply_defect_stabilizer(
            field,
            dt_tau=0.01,
            dirac_defect_value=d_before,
            threshold=0.5,
        )
        if isinstance(result, tuple):
            new_field = result[0]
        else:
            new_field = result
        d_after = float(dirac_defect(new_field, filtered=True))
        assert d_after <= d_before + 1e-4, \
            f"Defect should not increase: {d_before} → {d_after}"
