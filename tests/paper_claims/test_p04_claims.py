"""
P04 claim tests — Memory Coupling, Tau Cascade, and Virtual Population.

Run: python -m pytest -q _claude/tests/paper_claims/test_p04_claims.py
"""
import pytest
import jax.numpy as jnp
import numpy as np


# ── Helpers ─────────────────────────────────────────────────────────────

def _random_field(seed=42, H=16, W=16):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))


def _make_chunk(seed=7):
    """Minimal DocumentChunk for testing."""
    from ods_unified_v2.memory.models import DocumentChunk
    rng = np.random.default_rng(seed)
    return DocumentChunk(
        chunk_id=f"test-chunk-{seed}",
        doc_id="test-doc",
        chunk_index=0,
        version=1,
        source="test",
        text="test chunk for P04 claims",
        semantic_embedding=rng.normal(size=(384,)).astype(np.float32),
        spinor_fingerprint=rng.normal(size=(8,)).astype(np.float32),
    )


# ── Claim 1: Document injection alters field ─────────────────────────────

class TestDocumentInjection:

    @pytest.fixture
    def adapter(self):
        try:
            from ods_unified_v2.memory.field_adapter import inject_chunk, chunk_to_field_drive
            return {"inject": inject_chunk, "drive": chunk_to_field_drive}
        except ImportError:
            pytest.skip("memory.field_adapter not available")

    def test_injection_changes_field(self, adapter):
        Psi = _random_field()
        chunk = _make_chunk()
        Psi_after = adapter["inject"](Psi, chunk, strength=0.15)
        diff = float(jnp.sqrt(jnp.mean((Psi_after - Psi) ** 2)))
        assert diff > 1e-4, f"Injection should change field, got diff={diff}"

    def test_injection_scales_with_strength(self, adapter):
        Psi = _random_field()
        chunk = _make_chunk()
        diff_low = float(jnp.sqrt(jnp.mean((adapter["inject"](Psi, chunk, strength=0.05) - Psi) ** 2)))
        diff_high = float(jnp.sqrt(jnp.mean((adapter["inject"](Psi, chunk, strength=0.30) - Psi) ** 2)))
        assert diff_high > diff_low, \
            f"Higher strength should produce larger change: {diff_low} vs {diff_high}"

    def test_different_chunks_different_drives(self, adapter):
        chunk1 = _make_chunk(seed=7)
        chunk2 = _make_chunk(seed=99)
        drive1 = adapter["drive"](chunk1, (16, 16))
        drive2 = adapter["drive"](chunk2, (16, 16))
        cosine = float(jnp.sum(drive1 * drive2) / (jnp.linalg.norm(drive1) * jnp.linalg.norm(drive2) + 1e-8))
        assert cosine < 0.99, f"Different chunks should produce different drives, cosine={cosine}"


# ── Claim 2: Assignment energy ───────────────────────────────────────────

class TestAssignmentEnergy:

    @pytest.fixture
    def assim_module(self):
        try:
            from ods_unified_v2.jax_runtime.tau_assimilation import (
                assignment_energy,
                soft_assignment,
                make_source_meta,
                TauAssimilationConfig,
            )
            from ods_unified_v2.jax_runtime.oam_groups import extract_oam_groups
            return {
                "energy": assignment_energy,
                "soft": soft_assignment,
                "meta": make_source_meta,
                "config": TauAssimilationConfig,
                "groups": extract_oam_groups,
            }
        except ImportError:
            pytest.skip("tau_assimilation not available")

    def test_soft_assignment_sums_to_one(self, assim_module):
        rng = np.random.default_rng(42)
        energy = jnp.array(rng.uniform(size=(3, 5)).astype(np.float32))
        assignment = assim_module["soft"](energy, tau=0.2)
        row_sums = jnp.sum(assignment, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_assignment_energy_shape(self, assim_module):
        meta = assim_module["meta"](
            complexity=np.array([0.5, 0.8, 0.3]),
            bivector=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            resonance=np.array([0.6, 0.4, 0.7]),
        )
        # Need OAM groups — create a dummy 5D field
        rng = np.random.default_rng(42)
        field = jnp.array(rng.normal(size=(4, 8, 8, 8, 8)).astype(np.float32))
        groups = assim_module["groups"](field)
        cfg = assim_module["config"]()
        E = assim_module["energy"](meta, groups, cfg)
        assert E.shape[0] == 3, f"Expected 3 sources, got shape {E.shape}"
        assert E.shape[1] == groups.coherence.shape[0], f"Group dim mismatch"


# ── Claim 3: Coherence gating ───────────────────────────────────────────

class TestCoherenceGating:

    @pytest.fixture
    def cascade(self):
        try:
            from ods_unified_v2.lab.neuro_field.tau_cascade import coherence_gates
            from ods_unified_v2.lab.neuro_field.types import TauCascadeConfig
            return {"gates": coherence_gates, "config": TauCascadeConfig}
        except ImportError:
            pytest.skip("tau_cascade not available")

    def test_gate_closed_below_threshold(self, cascade):
        cfg = cascade["config"](coherence_vec_threshold=0.5, coherence_biv_threshold=0.3)
        result = cascade["gates"](0.2, 0.1, cfg)
        assert result["assoc_open"] is False
        assert result["bivector_stable"] is False

    def test_gate_open_above_threshold(self, cascade):
        cfg = cascade["config"](coherence_vec_threshold=0.5, coherence_biv_threshold=0.3)
        result = cascade["gates"](0.8, 0.6, cfg)
        assert result["assoc_open"] is True
        assert result["bivector_stable"] is True

    def test_gate_transition_at_threshold(self, cascade):
        cfg = cascade["config"](coherence_vec_threshold=0.5, coherence_biv_threshold=0.3)
        result_below = cascade["gates"](0.49, 0.29, cfg)
        result_above = cascade["gates"](0.51, 0.31, cfg)
        assert result_below["assoc_open"] is False
        assert result_above["assoc_open"] is True


# ── Claim 4: Virtual population layout ───────────────────────────────────

class TestVirtualPopulation:

    @pytest.fixture
    def pop_module(self):
        try:
            from ods_unified_v2.lab.neuro_field.virtual_population import (
                build_virtual_population_layout,
            )
            from ods_unified_v2.lab.neuro_field.types import MotifSpec
            return {"build": build_virtual_population_layout, "spec": MotifSpec}
        except ImportError:
            pytest.skip("virtual_population not available")

    def _motif(self, spec, label, sig, pair, grade):
        return spec(
            motif_id=label, label=label, kind="test", summary="test motif",
            preferred_pair=pair, preferred_grade=grade,
            spinor_signature=sig, center=(0.0, 0.0), sigma=1.0,
            polarity=1.0, tags=(),
        )

    def test_population_size(self, pop_module):
        motifs = [
            self._motif(pop_module["spec"], "m1", (1,0,0,0,0,0,0,0), (3, 6), 2),
            self._motif(pop_module["spec"], "m2", (0,1,0,0,0,0,0,0), (1, 4), 1),
        ]
        layout = pop_module["build"](motifs, replicas_per_motif=4, population_per_replica=2)
        expected = 2 * 4 * 2  # motifs × replicas × members
        assert int(layout.source_index.shape[0]) == expected

    def test_phase_offsets_unique(self, pop_module):
        motifs = [
            self._motif(pop_module["spec"], "m1", (1,0,0,0,0,0,0,0), (3, 6), 2),
        ]
        layout = pop_module["build"](motifs, replicas_per_motif=4, population_per_replica=3)
        offsets = np.array(layout.phase_offset)
        # All offsets should be unique
        assert len(np.unique(np.round(offsets, 6))) == len(offsets), "Phase offsets should be unique"


# ── Claim 5: CT free energy relaxation ───────────────────────────────────

class TestCTRelaxation:

    @pytest.fixture
    def relax_module(self):
        try:
            from ods_unified_v2.runtime.ct_relaxation import free_energy_12, relaxation_step_12
            return {"free_energy": free_energy_12, "step": relaxation_step_12}
        except ImportError:
            pytest.skip("ct_relaxation not available")

    def test_free_energy_computes(self, relax_module):
        v = np.array([0.5, 0.3, 0.1], dtype=np.float32)
        beta = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        fe = relax_module["free_energy"](v, beta)
        assert hasattr(fe, "F12")
        assert hasattr(fe, "mu1")
        assert hasattr(fe, "mu2")
        assert np.isfinite(fe.F12)

    def test_q_in_unit_interval(self, relax_module):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        beta = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        fe = relax_module["free_energy"](v, beta)
        assert 0 < fe.q < 1, f"q should be in (0,1), got {fe.q}"

    def test_c12_bounded(self, relax_module):
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = rng.normal(size=3).astype(np.float32)
            beta = rng.normal(size=3).astype(np.float32)
            fe = relax_module["free_energy"](v, beta)
            assert -1 - 1e-4 <= fe.c12 <= 1 + 1e-4, f"c12 should be in [-1,1], got {fe.c12}"
