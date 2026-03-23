"""
OVERNIGHT DATA COLLECTION — P03/P04/P05

Runs ~2-4 hours depending on hardware. Produces JSON results in
_claude/experiments/overnight_results/

Tests:
  1. Extended contractivity (10 seeds × 500 pairs × 5 grid sizes)
  2. Hutchinson convergence (50 iterations, track Hausdorff diameter)
  3. Commutator defect sweep (ω₁₂ × k_τ grid for P03)
  4. Texture multiescala stability (perturbation test)
  5. Texture regime separability (Thomas3D vs Spinor8D)
  6. Fixed point uniqueness (20 random starts, verify convergence)

Usage:
  cd /home/r/z_folder/a/conda/ods-unified-v2
  nohup python _claude/experiments/overnight_p05_data.py > _claude/experiments/overnight.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

# ── Setup ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "overnight_results"
RESULTS_DIR.mkdir(exist_ok=True)

def save_result(name, data):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"  → Saved {path}")


def timestamp():
    return time.strftime("%H:%M:%S")


# ── Map definitions (same as contractivity_test.py) ──────────────────────

def _scaled_rotation(Psi, r=0.9, theta=0.5):
    c, s = jnp.cos(theta), jnp.sin(theta)
    result = Psi.copy()
    v1, v2 = Psi[..., 1], Psi[..., 2]
    result = result.at[..., 1].set(c * v1 - s * v2)
    result = result.at[..., 2].set(s * v1 + c * v2)
    b1, b2 = Psi[..., 5], Psi[..., 6]
    result = result.at[..., 5].set(c * b1 - s * b2)
    result = result.at[..., 6].set(s * b1 + c * b2)
    return r * result


def _rotor_plus_delta(Psi, r=0.85, theta=0.3, delta=0.10):
    return _scaled_rotation(Psi, r=r, theta=theta) + delta * jnp.tanh(Psi)


def _field_norm(Psi):
    return float(jnp.sqrt(jnp.sum(jnp.square(Psi))))


def _contraction_ratio(Psi1, Psi2, map_fn):
    d_in = _field_norm(Psi1 - Psi2)
    if d_in < 1e-10:
        return float('nan')
    d_out = _field_norm(map_fn(Psi1) - map_fn(Psi2))
    return d_out / d_in


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Extended contractivity
# ══════════════════════════════════════════════════════════════════════════

def test_extended_contractivity():
    print(f"\n[{timestamp()}] TEST 1: Extended contractivity")
    print("=" * 70)

    seeds = list(range(10))  # 10 seeds
    n_pairs = 500
    grid_sizes = [(8, 8), (16, 16), (32, 32), (48, 48), (64, 64)]

    configs = [
        ("rotor_r090", 0.90, 0.00),
        ("rotor_r070", 0.70, 0.00),
        ("rotor_delta_085_010", 0.85, 0.10),
        ("rotor_delta_085_020", 0.85, 0.20),
        ("rotor_delta_070_025", 0.70, 0.25),
        ("rotor_delta_050_030", 0.50, 0.30),
        ("rotor_delta_090_015", 0.90, 0.15),  # bound = 1.05, near boundary
        ("rotor_delta_095_010", 0.95, 0.10),  # bound = 1.05, near boundary
    ]

    results = {}
    for name, r, delta in configs:
        map_fn = lambda P, r=r, d=delta: _rotor_plus_delta(P, r=r, delta=d) if d > 0 else _scaled_rotation(P, r=r)
        config_results = {}

        for H, W in grid_sizes:
            all_ratios = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                for _ in range(n_pairs):
                    Psi1 = jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))
                    Psi2 = jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))
                    c = _contraction_ratio(Psi1, Psi2, map_fn)
                    if not np.isnan(c):
                        all_ratios.append(c)

            arr = np.array(all_ratios)
            grid_key = f"{H}x{W}"
            config_results[grid_key] = {
                "n_samples": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "p99": float(np.percentile(arr, 99)),
                "p999": float(np.percentile(arr, 99.9)),
                "pct_below_1": float(np.mean(arr < 1.0)) * 100,
            }
            print(f"  {name:30s} {grid_key:6s}  "
                  f"mean={config_results[grid_key]['mean']:.5f}  "
                  f"max={config_results[grid_key]['max']:.5f}  "
                  f"p999={config_results[grid_key]['p999']:.5f}  "
                  f"%<1={config_results[grid_key]['pct_below_1']:.1f}%")

        results[name] = {
            "r": r, "delta": delta,
            "theoretical_bound": r + delta,
            "grids": config_results,
        }

    save_result("01_extended_contractivity", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Hutchinson convergence (extended)
# ══════════════════════════════════════════════════════════════════════════

def test_hutchinson_convergence():
    print(f"\n[{timestamp()}] TEST 2: Hutchinson convergence")
    print("=" * 70)

    H, W = 16, 16
    n_cloud = 80
    n_iterations = 50
    n_maps = 4
    thetas = [0.3, 0.7, 1.1, 1.9]

    configs = [
        ("conservative_r070_d010", 0.70, 0.10),
        ("moderate_r080_d015", 0.80, 0.15),
        ("aggressive_r085_d020", 0.85, 0.20),
        ("boundary_r090_d015", 0.90, 0.15),
    ]

    results = {}
    for label, r_val, delta_val in configs:
        print(f"  Config: {label}")
        rng = np.random.default_rng(42)
        cloud = [jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32)) for _ in range(n_cloud)]

        diameters = []
        ratios = []

        for iteration in range(n_iterations):
            new_cloud = []
            for Psi in cloud:
                for theta in thetas:
                    new_cloud.append(_rotor_plus_delta(Psi, r=r_val, theta=theta, delta=delta_val))

            if len(new_cloud) > n_cloud * 2:
                indices = rng.choice(len(new_cloud), size=n_cloud, replace=False)
                cloud = [new_cloud[i] for i in indices]
            else:
                cloud = new_cloud

            sample_size = min(40, len(cloud))
            sample_idx = rng.choice(len(cloud), size=sample_size, replace=False)
            max_dist = 0.0
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    d = _field_norm(cloud[sample_idx[i]] - cloud[sample_idx[j]])
                    if d > max_dist:
                        max_dist = d

            diameters.append(max_dist)
            if len(diameters) > 1 and diameters[-2] > 1e-8:
                ratios.append(max_dist / diameters[-2])

            if iteration % 10 == 0 or max_dist < 1e-4:
                ratio_str = f"  ratio={ratios[-1]:.4f}" if ratios else ""
                print(f"    iter {iteration:3d}  diameter={max_dist:.6f}{ratio_str}")

            if max_dist < 1e-6:
                print(f"    → CONVERGED at iteration {iteration}")
                break

        results[label] = {
            "r": r_val, "delta": delta_val,
            "diameters": diameters,
            "contraction_ratios": ratios,
            "final_diameter": diameters[-1],
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "converged": diameters[-1] < 1e-4,
            "iterations": len(diameters),
        }

    save_result("02_hutchinson_convergence", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Commutator defect sweep (P03 data)
# ══════════════════════════════════════════════════════════════════════════

def test_commutator_sweep():
    print(f"\n[{timestamp()}] TEST 3: Commutator defect sweep")
    print("=" * 70)

    from ods_unified_v2.jax_runtime.observability import compute_commutator_defect
    from ods_unified_v2.jax_runtime.integrator import default_params

    omega_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    ktau_values = [0, 1, 2, 3, 5, 8]
    seeds = [7, 17, 42]

    results = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        Psi = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
        seed_results = {}

        for omega in omega_values:
            for ktau in ktau_values:
                params = default_params()
                params["omega_12"] = omega
                params["gamma_12"] = omega * 0.67  # typical ratio
                params["k_tau"] = ktau
                params["tau_dt"] = 1e-3 if ktau > 0 else 0.0
                params["dt"] = 1e-2

                defect = compute_commutator_defect(Psi, params)
                key = f"w{omega:.2f}_k{ktau}"
                seed_results[key] = {
                    "omega": omega, "k_tau": ktau, "defect": defect,
                }

        results[f"seed_{seed}"] = seed_results
        print(f"  seed {seed}: {len(seed_results)} configurations measured")

    save_result("03_commutator_sweep", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Texture stability under perturbation
# ══════════════════════════════════════════════════════════════════════════

def test_texture_stability():
    print(f"\n[{timestamp()}] TEST 4: Texture stability")
    print("=" * 70)

    from ods_unified_v2.jax_runtime.gamma_virtual import gamma_focus_score, gamma_grade_entropy
    from ods_unified_v2.jax_runtime.amr import rotational_tension_map
    from ods_unified_v2.jax_runtime.metrics import runtime_metrics

    def texture_vector(Psi):
        """Simple texture: [focus, entropy, mean_tension, d_eff]."""
        focus = float(gamma_focus_score(Psi))
        entropy = float(gamma_grade_entropy(Psi))
        tension = float(jnp.mean(rotational_tension_map(Psi)))
        metrics = runtime_metrics(Psi)
        d_eff = float(metrics.get("d_eff", 0.0))
        return np.array([focus, entropy, tension, d_eff], dtype=np.float32)

    def multiscale_texture(Psi, T_fn, depth=5, rho=0.8):
        """Θ_ms = Σ ρᵏ · Θ(Tᵏ(Ψ))."""
        result = np.zeros(4, dtype=np.float32)
        current = Psi
        for k in range(depth):
            result += (rho ** k) * texture_vector(current)
            current = T_fn(current)
        return result

    def T_contraction(Psi):
        return 0.5 * _rotor_plus_delta(Psi, r=0.70, delta=0.10, theta=0.3) + \
               0.5 * _rotor_plus_delta(Psi, r=0.60, delta=0.05, theta=0.7)

    perturbation_scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    seeds = [7, 17, 42, 99, 123]
    results = {"single_scale": {}, "multi_scale": {}}

    for eps in perturbation_scales:
        single_diffs = []
        multi_diffs = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            Psi = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            delta = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32)) * eps
            Psi_pert = Psi + delta

            theta_base = texture_vector(Psi)
            theta_pert = texture_vector(Psi_pert)
            single_diffs.append(float(np.linalg.norm(theta_pert - theta_base)))

            theta_ms_base = multiscale_texture(Psi, T_contraction)
            theta_ms_pert = multiscale_texture(Psi_pert, T_contraction)
            multi_diffs.append(float(np.linalg.norm(theta_ms_pert - theta_ms_base)))

        results["single_scale"][str(eps)] = {
            "mean_diff": float(np.mean(single_diffs)),
            "std_diff": float(np.std(single_diffs)),
        }
        results["multi_scale"][str(eps)] = {
            "mean_diff": float(np.mean(multi_diffs)),
            "std_diff": float(np.std(multi_diffs)),
        }
        ratio = float(np.mean(multi_diffs)) / max(float(np.mean(single_diffs)), 1e-10)
        print(f"  ε={eps:.3f}  single={np.mean(single_diffs):.6f}  "
              f"multi={np.mean(multi_diffs):.6f}  ratio={ratio:.3f}")

    save_result("04_texture_stability", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Regime separability with texture
# ══════════════════════════════════════════════════════════════════════════

def test_regime_separability():
    print(f"\n[{timestamp()}] TEST 5: Regime separability")
    print("=" * 70)

    from ods_unified_v2.jax_runtime.gamma_virtual import gamma_focus_score, gamma_grade_entropy
    from ods_unified_v2.jax_runtime.amr import rotational_tension_map
    from ods_unified_v2.jax_runtime.metrics import runtime_metrics

    def texture_vector(Psi):
        focus = float(gamma_focus_score(Psi))
        entropy = float(gamma_grade_entropy(Psi))
        tension = float(jnp.mean(rotational_tension_map(Psi)))
        metrics = runtime_metrics(Psi)
        d_eff = float(metrics.get("d_eff", 0.0))
        return np.array([focus, entropy, tension, d_eff], dtype=np.float32)

    # Generate "Thomas3D-like" fields (low-D: only channels 1-3 active)
    # vs "Spinor8D-like" fields (all 8 channels active)
    n_samples = 200
    seeds = range(n_samples)

    thomas_textures = []
    spinor_textures = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Thomas-like: energy concentrated in vector channels
        thomas = np.zeros((16, 16, 8), dtype=np.float32)
        thomas[:, :, 1:4] = rng.normal(size=(16, 16, 3)).astype(np.float32)
        thomas[:, :, 0] = rng.normal(size=(16, 16)).astype(np.float32) * 0.1
        thomas_textures.append(texture_vector(jnp.array(thomas)))

        # Spinor-like: energy across all channels
        spinor = rng.normal(size=(16, 16, 8)).astype(np.float32)
        spinor_textures.append(texture_vector(jnp.array(spinor)))

    thomas_arr = np.array(thomas_textures)
    spinor_arr = np.array(spinor_textures)

    # Compute separability: inter-class distance / intra-class spread
    thomas_mean = np.mean(thomas_arr, axis=0)
    spinor_mean = np.mean(spinor_arr, axis=0)
    inter_dist = float(np.linalg.norm(thomas_mean - spinor_mean))

    thomas_spread = float(np.mean(np.linalg.norm(thomas_arr - thomas_mean, axis=1)))
    spinor_spread = float(np.mean(np.linalg.norm(spinor_arr - spinor_mean, axis=1)))
    intra_spread = (thomas_spread + spinor_spread) / 2

    separability = inter_dist / max(intra_spread, 1e-8)

    # Per-feature separability
    feature_names = ["focus", "entropy", "tension", "d_eff"]
    feature_sep = {}
    for i, name in enumerate(feature_names):
        t_mean = float(np.mean(thomas_arr[:, i]))
        s_mean = float(np.mean(spinor_arr[:, i]))
        t_std = float(np.std(thomas_arr[:, i]))
        s_std = float(np.std(spinor_arr[:, i]))
        pooled_std = np.sqrt((t_std**2 + s_std**2) / 2)
        cohens_d = abs(t_mean - s_mean) / max(pooled_std, 1e-8)
        feature_sep[name] = {
            "thomas_mean": t_mean, "spinor_mean": s_mean,
            "thomas_std": t_std, "spinor_std": s_std,
            "cohens_d": cohens_d,
        }
        print(f"  {name:12s}  Thomas={t_mean:.4f}±{t_std:.4f}  "
              f"Spinor={s_mean:.4f}±{s_std:.4f}  Cohen's d={cohens_d:.2f}")

    results = {
        "n_samples": n_samples,
        "overall_separability": separability,
        "inter_class_distance": inter_dist,
        "intra_class_spread": intra_spread,
        "features": feature_sep,
    }
    print(f"  Overall separability (inter/intra) = {separability:.3f}")

    save_result("05_regime_separability", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: Fixed point uniqueness (extended)
# ══════════════════════════════════════════════════════════════════════════

def test_fixed_point_uniqueness():
    print(f"\n[{timestamp()}] TEST 6: Fixed point uniqueness")
    print("=" * 70)

    def T(Psi):
        return 0.5 * _rotor_plus_delta(Psi, r=0.60, delta=0.10, theta=0.3) + \
               0.5 * _rotor_plus_delta(Psi, r=0.50, delta=0.05, theta=0.7)

    n_starts = 20
    n_iterations = 80
    H, W = 16, 16

    trajectories = []
    final_states = []

    for seed in range(n_starts):
        rng = np.random.default_rng(seed)
        Psi = jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))
        norms = []

        for step in range(n_iterations):
            Psi_next = T(Psi)
            diff = _field_norm(Psi_next - Psi)
            norms.append(diff)
            Psi = Psi_next

        final_states.append(Psi)
        trajectories.append({
            "seed": seed,
            "convergence_norms": norms,
            "final_diff": norms[-1],
        })

    # Pairwise distances between final states
    pairwise = []
    for i in range(len(final_states)):
        for j in range(i + 1, len(final_states)):
            d = _field_norm(final_states[i] - final_states[j])
            pairwise.append(d)

    pairwise_arr = np.array(pairwise)
    results = {
        "n_starts": n_starts,
        "n_iterations": n_iterations,
        "final_diffs": [t["final_diff"] for t in trajectories],
        "pairwise_final_distances": {
            "mean": float(np.mean(pairwise_arr)),
            "max": float(np.max(pairwise_arr)),
            "min": float(np.min(pairwise_arr)),
            "std": float(np.std(pairwise_arr)),
        },
        "all_converged_below_1e4": all(t["final_diff"] < 1e-4 for t in trajectories),
        "unique_fixed_point": float(np.max(pairwise_arr)) < 1e-3,
    }

    print(f"  {n_starts} starts × {n_iterations} iterations")
    print(f"  Final step diffs: max={max(results['final_diffs']):.6f}")
    print(f"  Pairwise final distances: mean={results['pairwise_final_distances']['mean']:.6f}  "
          f"max={results['pairwise_final_distances']['max']:.6f}")
    print(f"  Unique fixed point: {results['unique_fixed_point']}")

    save_result("06_fixed_point_uniqueness", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# TEST 7: Runtime maps contractivity (gamma_flow + defect_stabilizer)
# ══════════════════════════════════════════════════════════════════════════

def test_runtime_maps():
    print(f"\n[{timestamp()}] TEST 7: Runtime maps contractivity (extended)")
    print("=" * 70)

    from ods_unified_v2.jax_runtime.gamma_virtual import gamma_virtual_flow
    from ods_unified_v2.jax_runtime.defect_stabilizer import apply_defect_stabilizer
    from ods_unified_v2.jax_runtime.wilson import dirac_defect
    from ods_unified_v2.jax_runtime.operators import laplacian_2d

    def map_gamma(Psi, dt=0.01):
        flow, _ = gamma_virtual_flow(Psi)
        return Psi + dt * flow

    def map_defect(Psi, dt_tau=0.005):
        d = dirac_defect(Psi, filtered=True)
        result = apply_defect_stabilizer(Psi, dt_tau=dt_tau, dirac_defect_value=d)
        return result[0] if isinstance(result, tuple) else result

    def map_laplacian(Psi, alpha=0.05):
        return Psi + alpha * laplacian_2d(Psi)

    maps = {
        "gamma_flow_dt001": lambda P: map_gamma(P, dt=0.001),
        "gamma_flow_dt005": lambda P: map_gamma(P, dt=0.005),
        "gamma_flow_dt010": lambda P: map_gamma(P, dt=0.010),
        "gamma_flow_dt020": lambda P: map_gamma(P, dt=0.020),
        "defect_stab_dt001": lambda P: map_defect(P, dt_tau=0.001),
        "defect_stab_dt005": lambda P: map_defect(P, dt_tau=0.005),
        "defect_stab_dt010": lambda P: map_defect(P, dt_tau=0.010),
        "laplacian_a005": lambda P: map_laplacian(P, alpha=0.05),
        "laplacian_a010": lambda P: map_laplacian(P, alpha=0.10),
        "laplacian_a020": lambda P: map_laplacian(P, alpha=0.20),
    }

    seeds = list(range(10))
    n_pairs = 200
    results = {}

    for name, map_fn in maps.items():
        all_ratios = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for _ in range(n_pairs):
                Psi1 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
                Psi2 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
                try:
                    c = _contraction_ratio(Psi1, Psi2, map_fn)
                    if not np.isnan(c):
                        all_ratios.append(c)
                except Exception:
                    pass

        if all_ratios:
            arr = np.array(all_ratios)
            results[name] = {
                "n_samples": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "max": float(np.max(arr)),
                "p99": float(np.percentile(arr, 99)),
                "p999": float(np.percentile(arr, 99.9)),
                "pct_below_1": float(np.mean(arr < 1.0)) * 100,
            }
            print(f"  {name:25s}  mean={results[name]['mean']:.5f}  "
                  f"max={results[name]['max']:.5f}  %<1={results[name]['pct_below_1']:.1f}%")

    save_result("07_runtime_maps_contractivity", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(f"OVERNIGHT DATA COLLECTION — Started at {timestamp()}")
    print("=" * 70)
    t0 = time.time()

    try:
        test_commutator_sweep()         # ~5 min (P03 data)
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")

    try:
        test_texture_stability()        # ~10 min (P05 data)
    except Exception as e:
        print(f"  TEST 4 FAILED: {e}")

    try:
        test_regime_separability()      # ~5 min (P05 data)
    except Exception as e:
        print(f"  TEST 5 FAILED: {e}")

    try:
        test_fixed_point_uniqueness()   # ~10 min (P05 data)
    except Exception as e:
        print(f"  TEST 6 FAILED: {e}")

    try:
        test_runtime_maps()             # ~30 min (P05 data)
    except Exception as e:
        print(f"  TEST 7 FAILED: {e}")

    try:
        test_hutchinson_convergence()   # ~20 min (P05 data)
    except Exception as e:
        print(f"  TEST 2 FAILED: {e}")

    try:
        test_extended_contractivity()   # ~60-120 min (P05 data, heaviest)
    except Exception as e:
        print(f"  TEST 1 FAILED: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE — Total time: {elapsed/60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'=' * 70}")
