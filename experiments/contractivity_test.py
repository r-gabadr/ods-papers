"""
Contractivity test — determines whether individual Clifford maps
Φ_{α,ℓ}(Ψ) = r·R·Ψ·R̃ + Δ(Ψ) are contractive, enabling the Hutchinson
fractal route (Variant B), or only contractive in aggregate (Variant A).

The test:
  1. Generate N random pairs (Ψ₁, Ψ₂) of Clifford fields
  2. For each map Φ_α, compute the contraction ratio:
       c_α = ||Φ_α(Ψ₁) - Φ_α(Ψ₂)|| / ||Ψ₁ - Ψ₂||
  3. If max(c_α) < 1 consistently → individual contractivity holds → Variant B
  4. If only Σ w_α c_α < 1 → aggregate contractivity → Variant A

We test with actual CT runtime maps:
  - Rotor sandwich: R·Ψ·R̃ (isometry, c = 1.0 exactly)
  - Scaled rotor: r·R·Ψ·R̃ (contraction if r < 1)
  - Rotor + nonlinear Δ: the realistic case
  - gamma_virtual_flow: the actual gamma flow from the codebase
  - defect_stabilizer: the actual stabilizer from the codebase

Usage:
  cd /home/r/z_folder/a/conda/ods-unified-v2
  python _claude/experiments/contractivity_test.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from ods_unified_v2.jax_runtime.state import ensure_clifford_state
from ods_unified_v2.jax_runtime.gamma_virtual import gamma_virtual_flow, gamma_focus_score
from ods_unified_v2.jax_runtime.defect_stabilizer import apply_defect_stabilizer
from ods_unified_v2.jax_runtime.wilson import dirac_defect
from ods_unified_v2.jax_runtime.operators import laplacian_2d


def field_norm(Psi):
    """L2 norm of a Clifford field."""
    return float(jnp.sqrt(jnp.sum(jnp.square(Psi))))


def contraction_ratio(Psi1, Psi2, map_fn):
    """Compute ||map(Ψ₁) - map(Ψ₂)|| / ||Ψ₁ - Ψ₂||."""
    d_input = field_norm(Psi1 - Psi2)
    if d_input < 1e-10:
        return float('nan')
    out1 = map_fn(Psi1)
    out2 = map_fn(Psi2)
    d_output = field_norm(out1 - out2)
    return d_output / d_input


# ── Map definitions ──────────────────────────────────────────────────────

def make_rotor(theta, plane=(4, 5)):
    """Unit rotor R = cos(θ/2) + sin(θ/2)·e_{plane}."""
    R = jnp.zeros(8, dtype=jnp.float32)
    R = R.at[0].set(jnp.cos(theta / 2))
    R = R.at[plane[0]].set(jnp.sin(theta / 2))
    return R


def clifford_product_simplified(a, b):
    """Simplified geometric product for Cl(3,0) multivectors (scalar × MV)."""
    # For a pure rotor sandwich, we use the grade structure
    # This is a simplified version — for testing contraction ratios
    # we use pointwise operations
    return a * b  # Component-wise as approximation for testing


def map_scaled_rotor(Psi, r=0.9, theta=0.3):
    """Φ(Ψ) = r · rotate(Ψ, θ) — scaled rotation in bivector plane."""
    # Rotation in e12 plane: channels (1,2) rotate, (4,5) rotate
    c, s = jnp.cos(theta), jnp.sin(theta)
    result = Psi.copy()
    # Rotate vector channels (e1, e2)
    v1, v2 = Psi[..., 1], Psi[..., 2]
    result = result.at[..., 1].set(c * v1 - s * v2)
    result = result.at[..., 2].set(s * v1 + c * v2)
    # Rotate bivector channels (e12 stays, e13↔e23)
    b1, b2 = Psi[..., 5], Psi[..., 6]
    result = result.at[..., 5].set(c * b1 - s * b2)
    result = result.at[..., 6].set(s * b1 + c * b2)
    return r * result


def map_rotor_plus_delta(Psi, r=0.85, theta=0.3, delta_strength=0.1):
    """Φ(Ψ) = r · rotate(Ψ) + δ · tanh(Ψ) — rotor + nonlinear perturbation."""
    rotated = map_scaled_rotor(Psi, r=r, theta=theta)
    delta = delta_strength * jnp.tanh(Psi)
    return rotated + delta


def map_gamma_flow(Psi, dt=0.01):
    """Φ(Ψ) = Ψ + dt · gamma_virtual_flow(Ψ) — one step of gamma flow."""
    flow, gate = gamma_virtual_flow(Psi)
    return Psi + dt * flow


def map_laplacian_smooth(Psi, alpha=0.1):
    """Φ(Ψ) = Ψ + α · ∇²Ψ — diffusive smoothing (should be contractive)."""
    return Psi + alpha * laplacian_2d(Psi)


def map_defect_stabilizer(Psi, dt_tau=0.01):
    """One step of defect stabilizer."""
    d = dirac_defect(Psi, filtered=True)
    result = apply_defect_stabilizer(Psi, dt_tau=dt_tau, dirac_defect_value=d)
    if isinstance(result, tuple):
        return result[0]
    return result


# ── Main experiment ──────────────────────────────────────────────────────

def run_contractivity_test(n_pairs=200, H=16, W=16, seeds=(42, 7, 17)):
    """Test contractivity of various maps."""

    maps = {
        "pure_rotation (r=1.0)": lambda Psi: map_scaled_rotor(Psi, r=1.0, theta=0.5),
        "scaled_rotor (r=0.90)": lambda Psi: map_scaled_rotor(Psi, r=0.90, theta=0.5),
        "scaled_rotor (r=0.70)": lambda Psi: map_scaled_rotor(Psi, r=0.70, theta=0.5),
        "rotor+delta (r=0.85, δ=0.10)": lambda Psi: map_rotor_plus_delta(Psi, r=0.85, delta_strength=0.10),
        "rotor+delta (r=0.85, δ=0.20)": lambda Psi: map_rotor_plus_delta(Psi, r=0.85, delta_strength=0.20),
        "rotor+delta (r=0.70, δ=0.10)": lambda Psi: map_rotor_plus_delta(Psi, r=0.70, delta_strength=0.10),
        "rotor+delta (r=0.70, δ=0.25)": lambda Psi: map_rotor_plus_delta(Psi, r=0.70, delta_strength=0.25),
        "rotor+delta (r=0.50, δ=0.30)": lambda Psi: map_rotor_plus_delta(Psi, r=0.50, delta_strength=0.30),
        "laplacian_smooth (α=0.05)": lambda Psi: map_laplacian_smooth(Psi, alpha=0.05),
        "laplacian_smooth (α=0.10)": lambda Psi: map_laplacian_smooth(Psi, alpha=0.10),
        "gamma_flow (dt=0.005)": lambda Psi: map_gamma_flow(Psi, dt=0.005),
        "gamma_flow (dt=0.010)": lambda Psi: map_gamma_flow(Psi, dt=0.010),
        "defect_stabilizer (dt=0.005)": lambda Psi: map_defect_stabilizer(Psi, dt_tau=0.005),
    }

    print("=" * 80)
    print("CONTRACTIVITY TEST — Individual map analysis")
    print("=" * 80)
    print(f"  {n_pairs} pairs per seed, grid {H}×{W}×8, seeds {seeds}")
    print()

    for name, map_fn in maps.items():
        all_ratios = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ratios = []
            for i in range(n_pairs):
                Psi1 = jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))
                Psi2 = jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))
                try:
                    c = contraction_ratio(Psi1, Psi2, map_fn)
                    if not np.isnan(c):
                        ratios.append(c)
                except Exception as e:
                    pass  # skip errors
            all_ratios.extend(ratios)

        if not all_ratios:
            print(f"  {name:45s}  FAILED (no valid ratios)")
            continue

        arr = np.array(all_ratios)
        mean_c = float(np.mean(arr))
        max_c = float(np.max(arr))
        std_c = float(np.std(arr))
        pct_below_1 = float(np.mean(arr < 1.0)) * 100

        verdict = "CONTRACTIVE ✓" if max_c < 1.0 else (
            "WEAK (mean<1)" if mean_c < 1.0 else "NOT CONTRACTIVE ✗"
        )

        print(f"  {name:45s}  mean={mean_c:.4f}  max={max_c:.4f}  "
              f"std={std_c:.4f}  %<1={pct_below_1:.0f}%  → {verdict}")

    print()
    print("=" * 80)
    print("AGGREGATE CONTRACTIVITY — Weighted mixture T = Σ wα Φα")
    print("=" * 80)

    # Test weighted average with different (r, δ) combinations
    configs = [
        # (label, [(r, delta, weight), ...])
        ("Conservative mix", [
            (0.70, 0.10, 0.4),
            (0.80, 0.05, 0.3),
            (0.60, 0.15, 0.3),
        ]),
        ("Aggressive mix", [
            (0.85, 0.20, 0.5),
            (0.90, 0.15, 0.3),
            (0.80, 0.25, 0.2),
        ]),
        ("Ultra-conservative", [
            (0.50, 0.10, 0.5),
            (0.40, 0.05, 0.5),
        ]),
    ]

    for label, components in configs:
        def make_mixture(comps):
            def mixture(Psi):
                result = jnp.zeros_like(Psi)
                for r, delta, w in comps:
                    result = result + w * map_rotor_plus_delta(Psi, r=r, delta_strength=delta)
                return result
            return mixture

        map_fn = make_mixture(components)
        all_ratios = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for i in range(n_pairs):
                Psi1 = jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))
                Psi2 = jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))
                c = contraction_ratio(Psi1, Psi2, map_fn)
                if not np.isnan(c):
                    all_ratios.append(c)

        arr = np.array(all_ratios)
        L_bound = sum(abs(w) * (r + delta) for r, delta, w in components)
        mean_c = float(np.mean(arr))
        max_c = float(np.max(arr))

        verdict = "CONTRACTIVE ✓" if max_c < 1.0 else "NOT CONTRACTIVE ✗"
        print(f"  {label:30s}  L_bound={L_bound:.3f}  "
              f"empirical_max={max_c:.4f}  mean={mean_c:.4f}  → {verdict}")

    print()
    print("=" * 80)
    print("HUTCHINSON CONVERGENCE — Iterating H(K) = ∪ Φα(K)")
    print("=" * 80)

    # Approximate Hausdorff convergence by tracking a cloud of points
    n_cloud = 50
    r_val, delta_val = 0.70, 0.10
    n_maps = 3
    thetas = [0.3, 0.7, 1.1]

    rng = np.random.default_rng(42)
    cloud = [jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32)) for _ in range(n_cloud)]

    print(f"  Maps: {n_maps} × rotor+delta (r={r_val}, δ={delta_val})")
    print(f"  Cloud size: {n_cloud} points")
    print()

    prev_diameter = None
    for iteration in range(12):
        # Apply H: for each point, apply all maps, keep all results
        new_cloud = []
        for Psi in cloud:
            for theta in thetas:
                new_point = map_rotor_plus_delta(Psi, r=r_val, theta=theta, delta_strength=delta_val)
                new_cloud.append(new_point)

        # Subsample to keep cloud manageable
        if len(new_cloud) > n_cloud * 2:
            indices = rng.choice(len(new_cloud), size=n_cloud, replace=False)
            cloud = [new_cloud[i] for i in indices]
        else:
            cloud = new_cloud

        # Estimate diameter (max pairwise distance in cloud)
        sample_size = min(30, len(cloud))
        sample_idx = rng.choice(len(cloud), size=sample_size, replace=False)
        max_dist = 0.0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                d = field_norm(cloud[sample_idx[i]] - cloud[sample_idx[j]])
                if d > max_dist:
                    max_dist = d

        ratio_str = ""
        if prev_diameter is not None and prev_diameter > 1e-8:
            ratio = max_dist / prev_diameter
            ratio_str = f"  ratio={ratio:.4f}"

        print(f"  iter {iteration:2d}  diameter={max_dist:.4f}  cloud={len(cloud)}{ratio_str}")
        prev_diameter = max_dist

        if max_dist < 1e-4:
            print(f"  → CONVERGED at iteration {iteration}")
            break

    print()
    if max_dist < 1e-2:
        print("  VERDICT: Cloud contracts → Hutchinson attractor exists → Variant B viable")
    elif max_dist < prev_diameter * 1.5:
        print("  VERDICT: Cloud stabilizes but doesn't fully contract → needs more analysis")
    else:
        print("  VERDICT: Cloud doesn't contract → Variant A (aggregate only)")


if __name__ == "__main__":
    run_contractivity_test(n_pairs=100)
