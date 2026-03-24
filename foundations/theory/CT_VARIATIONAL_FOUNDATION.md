# CT Variational Foundation
## Calculus of Variations on Measure Spaces + Geometric Measure Theory

> This document is the mathematical blueprint for the CT framework.
> Everything in P01–P06 should eventually be derivable from here.

---

## 1. The Objects

### 1.1 Clifford Algebra Cl(p,q)

For signature (p,q), dimension n = p+q:
- Vector space of dimension 2^n
- Generators γ_1, ..., γ_n satisfying γ_μ γ_ν + γ_ν γ_μ = 2 g_{μν} I
- g = diag(+1,...,+1, -1,...,-1)  (p pluses, q minuses)
- Graded: Cl(p,q) = ⊕_{k=0}^{n} Cl^k(p,q)

### 1.2 Clifford-Valued Current

A k-current T on a domain Ω ⊂ R^n with values in Cl(p,q) is a continuous
linear functional on the space of smooth compactly-supported k-forms:

    T(ω) = ∫_M ⟨ω(x), T⃗(x)⟩ dH^k(x)

where:
- M is a k-rectifiable set (the support)
- T⃗(x) is a unit k-vector (blade) giving orientation at x
- H^k is the k-dimensional Hausdorff measure
- ⟨·,·⟩ is the Clifford inner product

**Key point**: The field Ψ in ODS is not "just an array". It is a section
of a Clifford bundle, and its singularities/defects are described by
currents in the GMT sense.

### 1.3 Mass of a Current

    M(T) = sup { T(ω) : |ω| ≤ 1 }

This is the GMT analogue of "total variation" or "energy". It measures
the geometric size of the current.

### 1.4 Boundary Operator

    (∂T)(ω) = T(dω)

where d is the exterior derivative. Stokes' theorem is built in:
∂∂ = 0.

---

## 2. The Slicing Theorem (The CT "Rebanada")

Given a current T and a Lipschitz function u: R^n → R^m, the slice
of T at value r is:

    ⟨T, u, r⟩ = ∂(T ⌞ {u < r}) - (∂T) ⌞ {u < r}

### 2.1 Coarea Formula

    ∫_{R^m} M(⟨T, u, r⟩) dr ≤ Lip(u)^m · M(T)

This is fundamental: it says the total mass of all slices is controlled
by the mass of the whole current times the Lipschitz constant.

### 2.2 CT Interpretation

Each Cl(p_k, q_k) in the signature ladder is a SLICE of the full
CT object. The function u that defines the slicing is the
**signature projection**:

    u: Ψ_CT → Ψ_k    (project to the k-th signature level)

The coarea formula guarantees that the total "information" across
slices is bounded by the total mass of the CT current.

### 2.3 Clifford Contraction for Slicing

In practice, the slice at level r is computed via left contraction:

    Slice_r(Ψ) = Ψ ⌋ δ(u(x) - r)

This is coordinate-free — it depends only on the intrinsic geometry
of the multivector.

---

## 3. The Action Functional

### 3.1 Dirichlet-Type Energy on Cl(p,q)

The fundamental energy functional for a Clifford-valued field Ψ is:

    S[Ψ] = ∫_Ω L(Ψ, DΨ) dμ

where:
- D = Σ_{μ=1}^{n} γ^μ ∂_μ   is the Dirac operator
- L is the Lagrangian density

The simplest physically meaningful Lagrangian is:

    L(Ψ, DΨ) = ½⟨DΨ, DΨ⟩ + V(Ψ)

where:
- ½⟨DΨ, DΨ⟩ = ½ Ψ† D†D Ψ   is the kinetic/gradient term
- V(Ψ) is the potential (self-interaction, constraints)

### 3.2 The Dual-Clock Extension

For the CT runtime with clocks t (fast) and τ (slow), the action
splits:

    S[Ψ] = S_t[Ψ] + S_τ[Ψ] + S_coupling[Ψ]

where:
    S_t[Ψ]  = ∫ ½|D_t Ψ|² + V_t(Ψ) dμ_t
    S_τ[Ψ]  = ∫ ½|D_τ Ψ|² + V_τ(Ψ) dμ_τ
    S_coupling = ∫ α(χ) · ⟨D_t Ψ, D_τ Ψ⟩ dμ

The handoff function α(χ) emerges as a **Lagrange multiplier**
or **penalty weight** that couples the two scales.

### 3.3 Euler-Lagrange Equations

    δS/δΨ = 0  ⟹  D†D Ψ + V'(Ψ) = 0

This is THE equation of motion. Everything in P01–P06 should be
a discretization or approximation of this.

### 3.4 D†D as Variational Operator

D†D is not just "Dirac squared". It is the **principal part** of
the Euler-Lagrange operator for the Dirichlet energy.

For Cl(p,q) with metric g:

    D†D = -Σ_{μ,ν} g^{μν} ∂_μ ∂_ν + lower order terms

This is:
- Elliptic when g is positive definite (Cl(n,0)) — diffusion/relaxation
- Hyperbolic when g has mixed signature (Cl(p,q), q>0) — wave propagation
- The "lower order terms" encode curvature, connection, torsion

### 3.5 Why This Matters for ODS

| ODS concept | Variational origin |
|---|---|
| χ_Cl (P03) | = ‖δS/δΨ‖ — residual of Euler-Lagrange |
| Conservation of E₁₂ (P04) | Noether theorem for transport symmetry |
| Handoff α (P03) | Lagrange multiplier in S_coupling |
| Contractivity (P05) | Coercivity of second variation δ²S > 0 |
| Solitons (P02) | Critical points of S with topological constraint |
| Closure breakdown (P06) | When the discretization of S fails to approximate the continuum |
| D†D spectral gap | Controls stability of critical points |

---

## 4. Noether's Theorem for CT

For every continuous symmetry of S[Ψ], there is a conserved current.

### 4.1 Translation Symmetry → Energy-Momentum

If S is invariant under x → x + a:

    ∂_μ T^{μν} = 0

where T^{μν} is the energy-momentum tensor of the Clifford field.

### 4.2 Grade Rotation Symmetry → E₁₂ Conservation

If S is invariant under grade rotations Ψ → e^{θB} Ψ e^{-θB}
for bivector B, and γ₁₂ = 0 (no dissipation):

    dE₁₂/dt = 0

This is what P04 tests empirically. With the variational framework,
it's a THEOREM, not an observation.

### 4.3 Scale Symmetry → τ-Clock Structure

If S has a discrete scale symmetry (self-similarity at discrete scales):

    S[Ψ(λx)] = λ^α S[Ψ(x)]

this generates the slow clock τ and the hierarchical structure
of the τ-cascade.

---

## 5. GMT for Defects and Singularities

### 5.1 Rectifiable Sets as Soliton Supports

A soliton in the ODS field is a concentration of energy along a
k-dimensional rectifiable set M_k. In GMT language:

    T_soliton = θ(x) · [[M_k]] ⊗ ξ(x)

where:
- M_k is the support (a rectifiable set)
- θ(x) is the multiplicity (energy density)
- ξ(x) is the Clifford orientation (k-vector)

### 5.2 Varifolds for Scale Structure

A varifold V is a measure on Ω × G(n,k) (domain × Grassmannian).
It captures not just WHERE the structure is, but its TANGENT PLANE
at each point.

For ODS, varifolds encode:
- Which grades are active at each point
- The local dimensionality of the field structure
- The effective dimension d_eff from P03

### 5.3 Blow-Up Analysis and Closure Breakdown

GMT provides tangent measures via blow-up:

    T_{x,r} = (η_{x,r})_# T / r^k

As r → 0, if T_{x,r} converges, the limit is a tangent cone.
If it doesn't converge → the point is a singularity.

This is the rigorous version of P06's closure breakdown:
χ_closure activates when the blow-up analysis fails to produce
a well-defined tangent.

### 5.4 Plateau Problem → Minimal CT Surfaces

The classic GMT problem: find a current T minimizing M(T) subject
to ∂T = B (given boundary).

In CT: given boundary conditions (observations, documents, priors),
find the field configuration Ψ that minimizes the action S[Ψ].
This is exactly what the runtime does iteratively.

---

## 6. The Full CT State (Signature Ladder)

### 6.1 Direct Sum Decomposition

    Ψ_CT = ⊕_{k ∈ {CT_slices}} ( Σ_I C_{k,I} Γ_{k,I} )

where for each slice k with signature (p_k, q_k):
- I is a multi-index over 2^{p_k+q_k} basis elements
- C_{k,I} are the coefficients (learnable/dynamic)
- Γ_{k,I} are the blades (structural)

### 6.2 Recursive Gamma Construction

    γ_new^(j) = γ_old^(j) ⊗ σ₃       (extend existing generators)
    γ_new^(n+1) = I ⊗ σ₁              (add positive-signature generator)
    γ_new^(n+2) = I ⊗ σ₂              (add negative-signature generator)

This gives a constructive procedure to build any Cl(p,q) from Cl(p-1,q)
or Cl(p,q-1), starting from Cl(0,0) = R.

### 6.3 Signature Ladder

| Level | Algebra | Dim | Generators | Physics / Role |
|-------|---------|-----|-----------|---------------|
| 0 | Cl(1,0) | 2 | γ₁ | Binary gate, sign, activation |
| 1 | Cl(2,0) | 4 | γ₁,γ₂ | Planar rotation, first bivector |
| 2 | Cl(3,0) | 8 | γ₁,γ₂,γ₃ | Spatial runtime (current ODS) |
| 3 | Cl(1,3) | 16 | γ₀,γ₁,γ₂,γ₃ | Spacetime, Lorentz, Dirac equation |
| 4 | Cl(4,1) | 32 | +γ₄,γ₅ | 3D conformal (CGA) |
| 5 | Cl(2,4) | 64 | ... | Spacetime conformal (CGA-ST) |

### 6.4 Inter-Slice Embedding

The embedding Cl(p,q) ↪ Cl(p',q') for p≤p', q≤q' is:

    ι(Ψ) = Ψ ⊗ 1_{extra dimensions}

The projection is:

    π(Ψ') = ⟨Ψ', 1_{extra}⟩ (trace out extra dimensions)

### 6.5 Action on the Full Ladder

    S_CT[Ψ_CT] = Σ_k w_k · S_k[Ψ_k] + Σ_{k<l} λ_{kl} · S_{coupling}[Ψ_k, Ψ_l]

where:
- S_k is the action on slice k
- S_coupling measures consistency between slices
- w_k, λ_{kl} are weights (possibly dynamic)

The coarea formula ensures: Σ_k M(Ψ_k) ≤ C · M(Ψ_CT)

---

## 7. From Theory to Code

### 7.1 Discretization Strategy

The continuous theory discretizes as:
- Ω → grid/graph (spatial discretization)
- H^k → counting measure or weighted graph measure
- D → finite difference Dirac (already in wilson.py)
- ∫ → sum over grid
- Currents → sparse arrays with support tracking

### 7.2 Module Map

| Theory | Module | Status |
|--------|--------|--------|
| Cl(p,q) constructor | ct_algebra.py | To build |
| Clifford currents | ct_currents.py | To build |
| Dirac D, D†D | ct_dirac.py | To build |
| Action S, Euler-Lagrange | ct_variational.py | To build |
| Embeddings ι, π | ct_embeddings.py | To build |
| Noether conservations | ct_noether.py | To build |
| Slicing ⟨T,u,r⟩ | ct_slicing.py | To build |

### 7.3 What P01–P06 Become

With this foundation:
- P01 (runtime) = discretization of S_t on Cl(3,0) slice
- P02 (regimes) = phase portrait of δS/δΨ = 0
- P03 (χ) = norm of Euler-Lagrange residual
- P04 (memory) = boundary conditions in the Plateau problem
- P05 (contractivity) = coercivity of δ²S
- P06 (closure) = failure of discretization to approximate S

---

## 8. Honest Scope

What this foundation DOES:
- Unifies all CT concepts under one variational principle
- Makes conservation laws theorems, not empirical observations
- Gives a rigorous meaning to "slicing" between signatures
- Connects to the full power of GMT for singularities/defects

What this foundation does NOT:
- Derive specific physical constants (c, ℏ, etc.)
- Prove that ODS fields model physical reality
- Guarantee that the discretization converges
- Replace the need for numerical experiments

The variational framework is the BLUEPRINT. The code is the BUILDING.
Both are needed.
