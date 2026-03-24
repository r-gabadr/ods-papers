"""
P06 claim tests — closure handoff and declarative control envelopes.

These tests stay deliberately modest: they verify the operational surrogates
that P06 is allowed to claim today.
"""

import numpy as np
import jax.numpy as jnp

from ods_unified_v2.jax_runtime.multiscale import regime_alpha, regime_blend, slice_consistency
from ods_unified_v2.jax_runtime.observability import (
    compute_chi,
    effective_mass_squared,
    handoff_alpha,
    tachyonic_instability_score,
)


def _random_field(seed=42, H=16, W=16):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))


class TestSmoothHandoff:
    def test_handoff_alpha_matches_regime_alpha(self):
        chi = jnp.array([0.0, 0.8, 1.0, 1.2, 2.0], dtype=jnp.float32)
        handoff = handoff_alpha(chi, chi0=1.0, epsilon=0.2)
        regime = regime_alpha(chi, chi0=1.0, epsilon=0.2)
        assert jnp.allclose(handoff, regime, atol=1e-6)

    def test_compute_chi_stays_small_when_all_breakdown_channels_are_zero(self):
        chi = float(
            compute_chi(
                eta=10.0,
                eta_crit=1.0,
                kn_loc=0.0,
                omega_mag=0.0,
                bkm_sentinel=0.0,
            )
        )
        assert chi < 1.0e-3, f"chi should stay small in the safe regime, got {chi}"

    def test_compute_chi_increases_with_mass_instability_channel(self):
        base = float(
            compute_chi(
                eta=10.0,
                eta_crit=1.0,
                kn_loc=0.0,
                omega_mag=0.0,
                bkm_sentinel=0.0,
                mass_instability=0.0,
                mass_weight=1.0,
            )
        )
        unstable = float(
            compute_chi(
                eta=10.0,
                eta_crit=1.0,
                kn_loc=0.0,
                omega_mag=0.0,
                bkm_sentinel=0.0,
                mass_instability=2.0,
                mass_weight=1.0,
            )
        )
        assert unstable > base


class TestShiftedInstabilityProxy:
    def test_effective_mass_squared_uses_shifted_low_mode(self):
        eigenvalues = jnp.array([0.2, 0.7, 1.1], dtype=jnp.float32)
        mass_sq = float(effective_mass_squared(eigenvalues, shift=0.5))
        assert abs(mass_sq + 0.3) < 1.0e-6

    def test_tachyonic_score_increases_when_mass_squared_turns_negative(self):
        stable = float(tachyonic_instability_score(0.2, scale=0.1))
        unstable = float(tachyonic_instability_score(-0.2, scale=0.1))
        assert unstable > stable


class TestBlendAndConsistency:
    def test_regime_blend_matches_endpoints(self):
        fast = _random_field(1)
        structural = _random_field(2)
        blended_fast = regime_blend(fast, structural, alpha=0.0, lambda_value=0.0)
        blended_struct = regime_blend(fast, structural, alpha=1.0, lambda_value=0.0)
        assert jnp.allclose(blended_fast, fast, atol=1.0e-6)
        assert jnp.allclose(blended_struct, structural, atol=1.0e-6)

    def test_slice_consistency_zero_for_identical_slices(self):
        base = _random_field(3)
        stacked = jnp.stack([base, base, base], axis=0)
        mismatch = float(slice_consistency(stacked))
        assert mismatch < 1.0e-6, f"identical slices should have ~0 mismatch, got {mismatch}"

    def test_slice_consistency_positive_for_distinct_slices(self):
        stacked = jnp.stack([_random_field(4), _random_field(5), _random_field(6)], axis=0)
        mismatch = float(slice_consistency(stacked))
        assert mismatch > 0.0
