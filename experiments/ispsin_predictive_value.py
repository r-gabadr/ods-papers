"""
Predictive value test for I_spin phase.

Question: does arg(I_spin) improve prediction of d_eff at +k ticks
         over baseline [q_OAM, H_grade, C_slice]?

Method (o4's suggestion):
  1. Generate N random Clifford fields
  2. Evolve each k steps with step_frozen
  3. Compute observables at t=0, target d_eff at t=k
  4. Ridge regression: X0=[C,H,q] vs X1=[C,H,q, Re(I_spin), Im(I_spin)]
  5. Compare out-of-sample MSE
  6. Permutation control: shuffle only arg(I_spin) → if gain disappears, phase was the key

If relative_gain > 0 stably across seeds → arg(I_spin) is a genuinely new observable.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import sys
sys.path.insert(0, "/home/r/z_folder/a/conda/ods-unified-v2/src")

from ods_unified_v2.jax_runtime.integrator import step_frozen
from ods_unified_v2.jax_runtime.oam_groups import oam_phase
from ods_unified_v2.jax_runtime.metrics import effective_dimension, grade_energies

jax.config.update("jax_enable_x64", False)

# ── observables ────────────────────────────────────────────────────────────────

def grade_entropy_norm(psi):
    d = grade_energies(psi)
    E = jnp.array([d['scalar'], d['vector'], d['bivector'], d['pseudoscalar']])
    p = E / (jnp.sum(E) + 1e-8)
    H = -jnp.sum(p * jnp.log(p + 1e-8))
    return H / jnp.log(4.0)   # normalized [0,1]

def bivector_phase_mean(psi):
    phi = jnp.arctan2(psi[..., 5], psi[..., 4])
    return jnp.angle(jnp.mean(jnp.exp(1j * phi.astype(jnp.complex64))))

def q_oam(psi):
    phi = jnp.arctan2(psi[..., 5], psi[..., 4])
    return jnp.abs(jnp.mean(jnp.exp(1j * phi.astype(jnp.complex64))))

def compute_observables(psi, psi_anchor):
    q   = float(q_oam(psi))
    H   = float(grade_entropy_norm(psi))
    C   = q - H                                      # focus score proxy
    deff = float(effective_dimension(psi))

    # I_spin components
    phi_i = float(bivector_phase_mean(psi))
    phi_a = float(bivector_phase_mean(psi_anchor))
    delta_theta = phi_i - phi_a
    mod   = q * (1.0 - H)
    re_i  = mod * float(jnp.cos(delta_theta))
    im_i  = mod * float(jnp.sin(delta_theta))

    return q, H, C, deff, re_i, im_i

# ── ridge regression ───────────────────────────────────────────────────────────

def ridge_fit(X, y, l2=1e-3):
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + l2 * I, X.T @ y)

def mse(X, y, w):
    return float(np.mean((X @ w - y) ** 2))

def eval_models(rows, k_future, split=0.8, n_perm=50, rng=None):
    rows = np.array(rows)
    # columns: q, H, C, deff_now, re_I, im_I, deff_future
    q, H, C, deff_now, re_I, im_I, deff_future = rows.T

    X0 = np.stack([q, H, C], axis=1)
    X1 = np.stack([q, H, C, re_I, im_I], axis=1)
    y  = deff_future

    n = len(y)
    k = int(split * n)

    X0tr, X0te = X0[:k], X0[k:]
    X1tr, X1te = X1[:k], X1[k:]
    ytr, yte   = y[:k],  y[k:]

    w0 = ridge_fit(X0tr, ytr)
    w1 = ridge_fit(X1tr, ytr)

    err0  = mse(X0te, yte, w0)
    err1  = mse(X1te, yte, w1)
    gain  = (err0 - err1) / (err0 + 1e-8)

    # Permutation control: shuffle only arg(I_spin)
    if rng is None:
        rng = np.random.default_rng(0)
    perm_gains = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(len(re_I[:k]))
        re_perm  = re_I[:k][perm_idx]
        im_perm  = im_I[:k][perm_idx]
        X1p_tr   = np.stack([q[:k], H[:k], C[:k], re_perm, im_perm], axis=1)
        wp        = ridge_fit(X1p_tr, ytr)
        perm_gains.append((err0 - mse(X1te, yte, wp)) / (err0 + 1e-8))

    perm_mean = float(np.mean(perm_gains))
    perm_std  = float(np.std(perm_gains))

    return {
        "mse_base":    err0,
        "mse_aug":     err1,
        "gain":        gain,
        "perm_mean":   perm_mean,
        "perm_std":    perm_std,
        "z_score":     (gain - perm_mean) / (perm_std + 1e-8),
    }

# ── main ───────────────────────────────────────────────────────────────────────

def run(N=800, H=16, W=16, k_future=5, split=0.8, seed=42):
    key = random.PRNGKey(seed)
    anchor_key, key = random.split(key)
    psi_anchor = random.normal(anchor_key, (H, W, 8)) * 0.5

    rows = []
    print(f"Generating {N} trajectories (k={k_future} steps)...")
    for i in range(N):
        key, subkey = random.split(key)
        psi0 = random.normal(subkey, (H, W, 8)) * 0.5

        # Evolve k steps
        psi_k = psi0
        for _ in range(k_future):
            psi_k = step_frozen(psi_k)

        q, H_obs, C, deff_now, re_i, im_i = compute_observables(psi0, psi_anchor)
        deff_future = float(effective_dimension(psi_k))

        rows.append([q, H_obs, C, deff_now, re_i, im_i, deff_future])

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N}")

    print("\nFitting ridge models...")
    results = eval_models(rows, k_future, split=split)

    print("\n── Predictive value test ───────────────────────────────────")
    print(f"  Baseline  [C, H, q]              MSE = {results['mse_base']:.5f}")
    print(f"  Augmented [C, H, q, Re, Im]      MSE = {results['mse_aug']:.5f}")
    print(f"  Relative gain                        = {results['gain']*100:.2f}%")
    print(f"  Permutation baseline (mean±std)      = {results['perm_mean']*100:.2f}% ± {results['perm_std']*100:.2f}%")
    print(f"  Z-score vs permutation               = {results['z_score']:.2f}")
    print()
    if results['z_score'] > 2.0:
        verdict = "PHASE IS NEW — arg(I_spin) improves prediction beyond chance (z>{:.1f})".format(results['z_score'])
    elif results['z_score'] > 1.0:
        verdict = "WEAK SIGNAL — marginal improvement, needs larger N or better target"
    else:
        verdict = "REDUNDANT — arg(I_spin) adds nothing beyond existing observables"
    print(f"  Verdict: {verdict}")

    return results

if __name__ == "__main__":
    run(N=800, k_future=5)
