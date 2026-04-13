"""
field_step.py — Campo compartido NS-viscoso diferenciable.
"""
from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp


class SharedField(eqx.Module):
    log_nu: jnp.ndarray
    log_alpha: jnp.ndarray
    dt: float = eqx.field(static=True)

    def __init__(self, nu: float = 0.1, alpha: float = 0.05, dt: float = 0.01):
        self.log_nu = jnp.log(jnp.array(nu))
        self.log_alpha = jnp.log(jnp.array(alpha))
        self.dt = dt

    @property
    def nu(self): return jnp.exp(self.log_nu)

    @property
    def alpha(self): return jnp.exp(self.log_alpha)

    def laplacian(self, Phi: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.roll(Phi, 1, 0) + jnp.roll(Phi, -1, 0) +
            jnp.roll(Phi, 1, 1) + jnp.roll(Phi, -1, 1) -
            4.0 * Phi
        )

    def step(self, Phi: jnp.ndarray) -> jnp.ndarray:
        return Phi + self.dt * (self.nu * self.laplacian(Phi) - self.alpha * Phi)

    def emit_to_field(self, Phi: jnp.ndarray, spinor: jnp.ndarray,
                      pos: tuple[int, int], sigma: jnp.ndarray) -> jnp.ndarray:
        """Splat gaussiano — pos son ints Python, sigma puede ser array JAX."""
        H, W, _ = Phi.shape
        rows = jnp.arange(H)
        cols = jnp.arange(W)
        rr, cc = jnp.meshgrid(rows, cols, indexing='ij')
        dist2 = (rr - pos[0])**2 + (cc - pos[1])**2
        weight = jnp.exp(-dist2 / (2 * sigma**2 + 1e-8))  # (H, W)
        return Phi + weight[..., None] * spinor[None, None, :]

    def sample(self, Phi: jnp.ndarray, pos: tuple[int, int]) -> jnp.ndarray:
        """Lee el campo en pos (ints Python). Devuelve (8,)."""
        r = pos[0] % Phi.shape[0]
        c = pos[1] % Phi.shape[1]
        return Phi[r, c]
