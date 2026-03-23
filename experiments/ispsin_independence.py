"""
Independence test: Is I_spin a genuinely new observable?

I_spin = q_OAM · (1 - H_grade/log4) · e^{iΔθ}

Hypothesis:
  - |I_spin| is redundant (linear combination of q_OAM * (1-H_grade))
  - arg(I_spin) = Δθ may carry new information (depends on anchor choice)

Method: Pearson correlation matrix over N random Clifford fields.
Variables: [q_OAM, H_grade, C_slice, d_eff, |I_spin|, cos(arg), sin(arg)]
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import sys
sys.path.insert(0, "/home/r/z_folder/a/conda/ods-unified-v2/src")

from ods_unified_v2.jax_runtime.oam_groups import oam_phase, group_coherence_mean, extract_oam_groups
from ods_unified_v2.jax_runtime.metrics import global_entropy, effective_dimension, grade_energies
from ods_unified_v2.jax_runtime.gamma_virtual import gamma_grade_entropy, gamma_focus_score

jax.config.update("jax_enable_x64", False)

# ── helpers ────────────────────────────────────────────────────────────────────

def grade_entropy(psi):
    """H over grade energy distribution. psi: [H, W, 8]"""
    d = grade_energies(psi)           # dict: scalar, vector, bivector, pseudoscalar
    E = jnp.array([d['scalar'], d['vector'], d['bivector'], d['pseudoscalar']])
    p = E / (jnp.sum(E) + 1e-8)
    return -jnp.sum(p * jnp.log(p + 1e-8))

def bivector_phase(psi):
    """φ_B = atan2(B₁₃, B₁₂) per spatial point. psi: [H, W, 8]"""
    return jnp.arctan2(psi[..., 5], psi[..., 4])   # B₁₃=idx5, B₁₂=idx4

def q_oam(psi):
    """Kuramoto order parameter on bivector phase."""
    phi = bivector_phase(psi)
    return jnp.abs(jnp.mean(jnp.exp(1j * phi.astype(jnp.complex64))))

def ispsin_components(psi, psi_anchor):
    """
    I_spin = q_OAM · (1 - H/log4) · e^{iΔθ}
    Returns: modulus, cos(arg), sin(arg)
    """
    q = q_oam(psi)
    H = grade_entropy(psi)
    phi_i = jnp.mean(bivector_phase(psi))
    phi_a = jnp.mean(bivector_phase(psi_anchor))
    delta_theta = phi_i - phi_a

    modulus = q * (1.0 - H / jnp.log(4.0))
    modulus = jnp.clip(modulus, 0.0, 1.0)

    return float(modulus), float(jnp.cos(delta_theta)), float(jnp.sin(delta_theta))

def clifford_slice(key, H=16, W=16):
    """Random Clifford field [H, W, 8]."""
    return random.normal(key, (H, W, 8))

def focus_score(psi):
    """gamma_focus_score proxy: coherence - entropy."""
    q = float(q_oam(psi))
    H = float(grade_entropy(psi)) / float(jnp.log(4.0))
    return q - H    # C_slice surrogate

# ── main experiment ────────────────────────────────────────────────────────────

def run(N=2000, H=16, W=16, seed=42):
    key = random.PRNGKey(seed)
    anchor_key, key = random.split(key)
    psi_anchor = clifford_slice(anchor_key, H, W)

    rows = []
    for i in range(N):
        key, subkey = random.split(key)
        psi = clifford_slice(subkey, H, W)

        q     = float(q_oam(psi))
        h     = float(grade_entropy(psi)) / float(jnp.log(4.0))   # normalized [0,1]
        deff  = float(effective_dimension(psi))
        c     = focus_score(psi)
        mod, cos_arg, sin_arg = ispsin_components(psi, psi_anchor)

        rows.append([q, h, c, deff, mod, cos_arg, sin_arg])

    X = np.array(rows)
    labels = ["q_OAM", "H_grade", "C_slice", "d_eff", "|I_spin|", "cos(arg)", "sin(arg)"]

    corr = np.corrcoef(X.T)

    print("\n── Pearson correlation matrix ──────────────────────────────")
    header = f"{'':>10}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    for i, li in enumerate(labels):
        row = f"{li:>10}" + "".join(f"{corr[i,j]:>10.3f}" for j in range(len(labels)))
        print(row)

    # Key findings
    print("\n── Key findings ────────────────────────────────────────────")
    idx = {l: i for i, l in enumerate(labels)}

    r_mod_q  = corr[idx["|I_spin|"], idx["q_OAM"]]
    r_mod_h  = corr[idx["|I_spin|"], idx["H_grade"]]
    r_cos_q  = corr[idx["cos(arg)"], idx["q_OAM"]]
    r_sin_q  = corr[idx["sin(arg)"], idx["q_OAM"]]
    r_cos_h  = corr[idx["cos(arg)"], idx["H_grade"]]

    print(f"  |I_spin| vs q_OAM    : r = {r_mod_q:.3f}  {'(REDUNDANT)' if abs(r_mod_q)>0.9 else '(independent)'}")
    print(f"  |I_spin| vs H_grade  : r = {r_mod_h:.3f}  {'(REDUNDANT)' if abs(r_mod_h)>0.9 else '(independent)'}")
    print(f"  cos(arg) vs q_OAM    : r = {r_cos_q:.3f}  {'(REDUNDANT)' if abs(r_cos_q)>0.9 else '(independent)'}")
    print(f"  sin(arg) vs q_OAM    : r = {r_sin_q:.3f}  {'(REDUNDANT)' if abs(r_sin_q)>0.9 else '(independent)'}")
    print(f"  cos(arg) vs H_grade  : r = {r_cos_h:.3f}  {'(REDUNDANT)' if abs(r_cos_h)>0.9 else '(independent)'}")

    # Rank of full matrix
    rank = np.linalg.matrix_rank(corr, tol=0.1)
    print(f"\n  Effective rank (tol=0.1): {rank} / {len(labels)}")
    print(f"  → {'Some variables ARE redundant' if rank < len(labels) else 'All variables independent'}")

    return corr, labels

if __name__ == "__main__":
    run(N=2000)
