"""
expert.py — Un experto ODS como equinox.Module.

Un Expert es un agente con:
  - rotor:     R en Cl(3,0), transforma su spinor local
  - theta:     umbral de coherencia — cuando habla
  - sigma:     radio de emision al campo compartido
  - w_out:     proyeccion lineal 8->output_dim

El Expert NO sabe nada del campo global. Solo lee su entorno local
(un vector de 8 componentes) y decide si emitir o no.
"""
from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp


class Expert(eqx.Module):
    rotor: jnp.ndarray      # (8,)  — multivector en Cl(3,0)
    w_out: jnp.ndarray      # (8, out_dim) — proyeccion de salida
    log_theta: jnp.ndarray  # () — log del umbral (siempre positivo)
    log_sigma: jnp.ndarray  # () — log del radio de emision

    def __init__(self, out_dim: int, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        base = jnp.zeros(8).at[0].set(1.0)
        self.rotor = base + jax.random.normal(k1, (8,)) * 0.02
        self.w_out = jax.random.normal(k2, (8, out_dim)) * 0.1
        self.log_theta = jnp.log(jnp.array(0.4))
        self.log_sigma = jnp.log(jnp.array(1.0))

    @property
    def theta(self) -> jnp.ndarray:
        return jnp.exp(self.log_theta)

    @property
    def sigma(self) -> jnp.ndarray:
        return jnp.exp(self.log_sigma)

    def think(self, local_field: jnp.ndarray) -> jnp.ndarray:
        """Rota el campo local con el rotor propio. Devuelve spinor interno."""
        psi = self.rotor * local_field
        return psi / (jnp.linalg.norm(psi) + 1e-8)

    def coherence(self, psi: jnp.ndarray) -> jnp.ndarray:
        """vec_coherence = ||v|| / (||v|| + ||B|| + eps)"""
        v = jnp.linalg.norm(psi[1:4])
        b = jnp.linalg.norm(psi[4:7])
        return v / (v + b + 1e-8)

    def emit(self, psi: jnp.ndarray) -> tuple:
        """Gate diferenciable + proyeccion de salida."""
        coh = self.coherence(psi)
        gate = jax.nn.sigmoid((coh - self.theta) / 0.05)
        output = gate * (psi @ self.w_out)
        return output, gate

    def __call__(self, local_field: jnp.ndarray) -> tuple:
        psi = self.think(local_field)
        return self.emit(psi)
