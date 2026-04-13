"""
moe.py — CoherenceMoE: MoE gateado por coherencia sobre campo compartido.
"""
from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
from .expert import Expert
from .field_step import SharedField


class CoherenceMoE(eqx.Module):
    experts: list
    field: SharedField
    w_in: jnp.ndarray      # (d_model, 8)
    w_merge: jnp.ndarray   # (d_model, d_model)
    # Posiciones como lista Python de ints — estaticas, no trazadas
    _positions: list = eqx.field(static=True)
    _grid_size: int = eqx.field(static=True)

    def __init__(self, n_experts: int, d_model: int, grid_size: int = 16, key: jax.Array = None):
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, n_experts + 2)

        self.experts = [Expert(out_dim=d_model, key=keys[i]) for i in range(n_experts)]
        self.field = SharedField(nu=0.1, alpha=0.05, dt=0.01)
        self.w_in = jax.random.normal(keys[-2], (d_model, 8)) * 0.1
        self.w_merge = jax.random.normal(keys[-1], (d_model, d_model)) * 0.1
        self._grid_size = grid_size

        # Posiciones como lista Python de tuplas de ints
        n = int(jnp.ceil(jnp.sqrt(n_experts)))
        step = max(1, grid_size // (n + 1))
        positions = []
        for idx in range(n_experts):
            r = (idx // n) * step + step
            c = (idx % n) * step + step
            positions.append((min(r, grid_size - 1), min(c, grid_size - 1)))
        self._positions = positions

    def __call__(self, x: jnp.ndarray, field_state: jnp.ndarray = None) -> tuple:
        G = self._grid_size
        if field_state is None:
            field_state = jnp.zeros((G, G, 8))

        # 1. Proyectar token al espacio Cl(3,0)
        token_spinor = x @ self.w_in  # (8,)

        # 2. Emitir token al campo (posicion central)
        Phi = self.field.emit_to_field(field_state, token_spinor, (G // 2, G // 2), sigma=3.0)

        # 3. Cada experto lee, piensa, emite
        outputs = []
        gates = []
        for i, expert in enumerate(self.experts):
            pos = self._positions[i]
            local = self.field.sample(Phi, pos)  # (8,)
            out, gate = expert(local)
            outputs.append(out)
            gates.append(gate)
            psi = expert.think(local)
            Phi = self.field.emit_to_field(Phi, psi * gate, pos, sigma=expert.sigma)

        # 4. Campo evoluciona
        Phi = self.field.step(Phi)

        # 5. Mezcla ponderada por gates
        gates_arr = jnp.stack(gates)
        outputs_arr = jnp.stack(outputs)
        total_gate = jnp.sum(gates_arr) + 1e-8
        merged = jnp.sum(gates_arr[:, None] * outputs_arr, axis=0) / total_gate

        # 6. Proyeccion final
        output = merged @ self.w_merge

        return output, Phi, gates_arr
