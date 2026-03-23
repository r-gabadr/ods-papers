"""
Predictive value test v2 — phase-sensitive targets.

d_eff was the wrong target (energetic, already captured by [C,H,q]).
Phase information should predict phase-sensitive future quantities:

  Target A: q_OAM at t+k        (future coherence)
  Target B: bivector_phase_mean at t+k  (future phase direction)
  Target C: arg(I_spin) at t+k  (does current phase predict future phase?)

If arg(I_spin) improves prediction of any phase target → genuinely new.
Also tests k=5 vs k=10 (does signal persist over longer horizons?).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import sys
sys.path.insert(0, "/home/r/z_folder/a/conda/ods-unified-v2/src")

from ods_unified_v2.jax_runtime.integrator import step_frozen
from ods_unified_v2.jax_runtime.metrics import effective_dimension, grade_energies

jax.config.update("jax_enable_x64", False)

# ── observables ────────────────────────────────────────────────────────────────

def grade_entropy_norm(psi):
    d = grade_energies(psi)
    E = jnp.array([d['scalar'], d['vector'], d['bivector'], d['pseudoscalar']])
    p = E / (jnp.sum(E) + 1e-8)
    return float(-jnp.sum(p * jnp.log(p + 1e-8)) / jnp.log(4.0))

def q_oam(psi):
    phi = jnp.arctan2(psi[..., 5], psi[..., 4])
    return float(jnp.abs(jnp.mean(jnp.exp(1j * phi.astype(jnp.complex64)))))

def phase_mean(psi):
    phi = jnp.arctan2(psi[..., 5], psi[..., 4])
    return float(jnp.angle(jnp.mean(jnp.exp(1j * phi.astype(jnp.complex64)))))

def ispsin_delta(psi, psi_anchor):
    phi_i = phase_mean(psi)
    phi_a = phase_mean(psi_anchor)
    delta  = phi_i - phi_a
    q      = q_oam(psi)
    H      = grade_entropy_norm(psi)
    mod    = q * (1.0 - H)
    return mod * float(jnp.cos(delta)), mod * float(jnp.sin(delta)), delta

def evolve(psi, k):
    for _ in range(k):
        psi = step_frozen(psi)
    return psi

# ── ridge ──────────────────────────────────────────────────────────────────────

def ridge_fit(X, y, l2=1e-3):
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + l2 * I, X.T @ y)

def mse(X, y, w):
    return float(np.mean((X @ w - y) ** 2))

def eval_pair(X0, X1, y, split=0.8, n_perm=100, rng=None):
    n = len(y); k = int(split * n)
    X0tr, X0te = X0[:k], X0[k:]
    X1tr, X1te = X1[:k], X1[k:]
    ytr, yte   = y[:k],  y[k:]

    w0 = ridge_fit(X0tr, ytr)
    w1 = ridge_fit(X1tr, ytr)
    err0 = mse(X0te, yte, w0)
    err1 = mse(X1te, yte, w1)
    gain = (err0 - err1) / (err0 + 1e-8)

    if rng is None:
        rng = np.random.default_rng(0)
    perm_gains = []
    for _ in range(n_perm):
        idx = rng.permutation(k)
        Xp  = np.column_stack([X1tr[:, :3], X1tr[idx, 3:]])
        wp  = ridge_fit(Xp, ytr)
        perm_gains.append((err0 - mse(X1te, yte, wp)) / (err0 + 1e-8))

    pm = float(np.mean(perm_gains))
    ps = float(np.std(perm_gains))
    z  = (gain - pm) / (ps + 1e-8)
    return dict(err0=err0, err1=err1, gain=gain, perm_mean=pm, perm_std=ps, z=z)

# ── main ───────────────────────────────────────────────────────────────────────

def run(N=1200, grid=16, seeds=(42, 7, 17), ks=(5, 10)):
    rng = np.random.default_rng(0)

    for seed in seeds:
        key = random.PRNGKey(seed)
        anchor_key, key = random.split(key)
        psi_anchor = random.normal(anchor_key, (grid, grid, 8)) * 0.5

        print(f"\n{'='*60}")
        print(f"Seed {seed} — {N} trajectories")
        print(f"{'='*60}")

        # collect data for both k values at once
        data = {k: [] for k in ks}
        k_max = max(ks)

        for i in range(N):
            key, subkey = random.split(key)
            psi0 = random.normal(subkey, (grid, grid, 8)) * 0.5

            q0   = q_oam(psi0)
            H0   = grade_entropy_norm(psi0)
            C0   = q0 - H0
            re0, im0, delta0 = ispsin_delta(psi0, psi_anchor)

            # evolve step by step, record at each k
            psi = psi0
            for step_i in range(1, k_max + 1):
                psi = step_frozen(psi)
                if step_i in ks:
                    q_k     = q_oam(psi)
                    phi_k   = phase_mean(psi)
                    _, _, delta_k = ispsin_delta(psi, psi_anchor)
                    data[step_i].append([q0, H0, C0, re0, im0, q_k, phi_k, delta_k])

            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{N}")

        for k in ks:
            arr = np.array(data[k])
            q0, H0, C0, re0, im0 = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
            q_k, phi_k, delta_k  = arr[:,5], arr[:,6], arr[:,7]

            X0 = np.stack([q0, H0, C0], axis=1)
            X1 = np.stack([q0, H0, C0, re0, im0], axis=1)

            print(f"\n  k = +{k} steps")
            print(f"  {'Target':<28} {'gain':>8}  {'z-score':>8}  {'verdict'}")
            print(f"  {'-'*65}")

            for target_name, y in [("q_OAM future", q_k),
                                    ("phase_mean future", phi_k),
                                    ("arg(I_spin) future", delta_k)]:
                r = eval_pair(X0, X1, y, rng=rng)
                v = "NEW ✓" if r['z'] > 2.0 else ("weak" if r['z'] > 1.0 else "redundant")
                print(f"  {target_name:<28} {r['gain']*100:>7.2f}%  {r['z']:>8.2f}  {v}")

if __name__ == "__main__":
    run(N=1200, grid=16, seeds=(42, 7, 17), ks=(5, 10))
