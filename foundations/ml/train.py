"""
train.py — Loop de entrenamiento para CoherenceMoE.

Toy task: regresion sobre embeddings aleatorios.
(Sustituir por cualquier tarea real: language modeling, clasificacion, etc.)

Metricas que registra:
  - loss (MSE)
  - mean_gate: que fraccion de expertos hablo en media
  - gate_entropy: diversidad de uso de expertos
  - field_energy: energia total del campo compartido
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from .moe import CoherenceMoE


def make_toy_batch(key: jax.Array, batch: int, d_model: int):
    """Batch sintetico: x -> y = relu(Wx) con W aleatorio fijo."""
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (d_model, d_model)) * 0.3
    x = jax.random.normal(k2, (batch, d_model))
    y = jax.nn.relu(x @ W)
    return x, y


def loss_fn(model: CoherenceMoE, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """MSE sobre el batch. Procesa cada token secuencialmente (sin batch paralelo todavia)."""
    total_loss = 0.0
    for i in range(x.shape[0]):
        out, _, _ = model(x[i])
        total_loss += jnp.mean((out - y[i])**2)
    return total_loss / x.shape[0]


@eqx.filter_jit
def train_step(model: CoherenceMoE, opt_state, x, y, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(
    n_experts: int = 4,
    d_model: int = 32,
    n_steps: int = 200,
    batch_size: int = 8,
    lr: float = 3e-4,
    seed: int = 1337,
):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)

    model = CoherenceMoE(n_experts=n_experts, d_model=d_model, key=mk)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print(f"CoherenceMoE: {n_experts} experts, d_model={d_model}, lr={lr}")
    print(f"Params: {sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))} total floats")
    print()

    for step in range(n_steps):
        key, bk = jax.random.split(key)
        x, y = make_toy_batch(bk, batch_size, d_model)
        model, opt_state, loss = train_step(model, opt_state, x, y, optimizer)

        if step % 20 == 0:
            # Diagnosticos adicionales
            out, Phi, gates = model(x[0])
            field_energy = float(jnp.mean(Phi**2))
            mean_gate = float(jnp.mean(gates))
            # Entropia de uso de expertos
            g = gates / (jnp.sum(gates) + 1e-8)
            entropy = float(-jnp.sum(g * jnp.log(g + 1e-8)))
            print(f"step {step:4d} | loss {float(loss):.4f} | gate {mean_gate:.3f} | H_gate {entropy:.3f} | E_field {field_energy:.5f}")

    print()
    print("Thetas finales (umbrales de coherencia por experto):")
    for i, e in enumerate(model.experts):
        print(f"  expert {i}: theta={float(e.theta):.4f}  sigma={float(e.sigma):.4f}")
    return model


if __name__ == "__main__":
    train()
