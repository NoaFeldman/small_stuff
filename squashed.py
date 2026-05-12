"""Variational upper bound on the squashed entanglement (JAX backend).

E_sq(A:B) := min_{rho_ABE : Tr_E rho_ABE = rho_AB}  I(A:B|E),

with I(A:B|E) = S(rho_AE) + S(rho_BE) - S(rho_E) - S(rho_ABE).

We parametrize all valid extensions by an isometry  V : R -> E (x) E_aux,
where R is the purifying reference (NR = rank(rho_AB)) and E_aux has
dimension NR.  V is obtained from the QR factor of an unconstrained complex
matrix W of shape (NE * NR,  NR), so the optimization is unconstrained.

Backend: JAX (analytic gradient via reverse-mode autodiff, JIT-compiled)
plus scipy.optimize L-BFGS-B.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


_LOG_EPS = 1e-12


def _purify(rho: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, int]:
    """Return matrix psi of shape (D, NR) with psi @ psi^dagger = rho."""
    w, U = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    w = np.clip(w.real, 0.0, None)
    mask = w > tol
    w = w[mask]
    U = U[:, mask]
    if w.size == 0:
        raise ValueError("rho is (numerically) zero.")
    return U * np.sqrt(w), int(w.size)


def _entropy(rho: jnp.ndarray, log_base: float) -> jnp.ndarray:
    rho_h = 0.5 * (rho + rho.conj().T)
    w = jnp.linalg.eigvalsh(rho_h)
    w = jnp.clip(w.real, 0.0, None)
    safe = jnp.where(w > _LOG_EPS, w, 1.0)
    log_w = jnp.log(safe) / log_base
    return -jnp.sum(jnp.where(w > _LOG_EPS, w * log_w, 0.0))


def _build_objective(psi_mat: np.ndarray, NA: int, NB: int, NE: int, NR: int,
                     base: float):
    NE_aux = NR
    rows = NE * NE_aux
    n_complex = rows * NR
    log_base = float(jnp.log(base))
    psi_j = jnp.asarray(psi_mat)

    def cmi(x: jnp.ndarray) -> jnp.ndarray:
        W = (x[:n_complex] + 1j * x[n_complex:]).reshape(rows, NR)
        V, _ = jnp.linalg.qr(W)                         # (rows, NR), V^dag V = I
        psi_p = psi_j @ V.T                             # (D_AB, rows)
        Psi = psi_p.reshape(NA, NB, NE, NE_aux)
        T = jnp.einsum('abek,ABEk->abeABE', Psi, Psi.conj())
        D = NA * NB * NE
        rho_ABE = T.reshape(D, D)
        rho_AE = jnp.einsum('abeAbE->aeAE', T).reshape(NA * NE, NA * NE)
        rho_BE = jnp.einsum('abeaBE->beBE', T).reshape(NB * NE, NB * NE)
        rho_E = jnp.einsum('abeabE->eE', T).reshape(NE, NE)
        return (_entropy(rho_AE, log_base) + _entropy(rho_BE, log_base)
                - _entropy(rho_E, log_base) - _entropy(rho_ABE, log_base))

    val_grad = jax.jit(jax.value_and_grad(cmi))

    def f_and_g(x_np: np.ndarray):
        v, g = val_grad(jnp.asarray(x_np))
        return float(np.asarray(v).real), np.asarray(g).astype(np.float64)

    return f_and_g, 2 * n_complex


def _build_rho_ABE(x: np.ndarray, psi_mat: np.ndarray, NA: int, NB: int,
                   NE: int, NR: int) -> np.ndarray:
    NE_aux = NR
    rows = NE * NE_aux
    n = rows * NR
    W = (x[:n] + 1j * x[n:]).reshape(rows, NR)
    V, _ = np.linalg.qr(W)
    psi_p = psi_mat @ V.T
    Psi = psi_p.reshape(NA * NB, NE, NE_aux)
    rho_tensor = np.einsum('aek,bfk->aebf', Psi, Psi.conj())
    return rho_tensor.reshape(NA * NB * NE, NA * NB * NE)


def squashed_entanglement_upper_bound(
    rho_AB: np.ndarray,
    NA: int,
    NB: int,
    NE: int,
    *,
    base: float = 2.0,
    n_restarts: int = 8,
    max_iter: int = 2000,
    seed: int | None = None,
    x0: np.ndarray | None = None,
    return_x: bool = False,
    verbose: bool = False,
):
    """Variational upper bound on E_sq(A:B) using an analytic JAX gradient.

    Parameters
    ----------
    rho_AB : (NA*NB, NA*NB) density matrix.
    NA, NB, NE : Hilbert-space dimensions.
    n_restarts : random restarts in addition to the warm start (if any).
    x0 : optional warm-start parameter vector.
    return_x : if True, also return the optimal parameter vector.

    Returns
    -------
    (cmi, rho_ABE)              if return_x is False
    (cmi, rho_ABE, x_optimal)   if return_x is True
    """
    rng = np.random.default_rng(seed)
    D_AB = NA * NB
    if rho_AB.shape != (D_AB, D_AB):
        raise ValueError(f"rho_AB has shape {rho_AB.shape}, expected ({D_AB}, {D_AB}).")

    psi_mat, NR = _purify(rho_AB)
    f_and_g, n_params = _build_objective(psi_mat, NA, NB, NE, NR, base)

    best_val = np.inf
    best_x: np.ndarray | None = None

    def run(x_init: np.ndarray, label: str) -> None:
        nonlocal best_val, best_x
        res = minimize(f_and_g, x_init, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-12})
        if verbose:
            print(f"  {label}: CMI = {res.fun:.6f}  iters={res.nit}  success={res.success}")
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = res.x

    if x0 is not None:
        if x0.size != n_params:
            if verbose:
                print(f"  (warm-start size {x0.size} != {n_params}, ignored)")
        else:
            run(x0.astype(np.float64), "warm")

    for k in range(n_restarts):
        run(rng.standard_normal(n_params) * 0.5, f"restart {k + 1}/{n_restarts}")

    assert best_x is not None
    rho_ABE = _build_rho_ABE(best_x, psi_mat, NA, NB, NE, NR)

    if return_x:
        return best_val, rho_ABE, best_x
    return best_val, rho_ABE


if __name__ == "__main__":
    # Quick smoke test.
    rng = np.random.default_rng(0)
    NA = NB = NE = 2
    d = NA * NB
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = X @ X.conj().T
    rho /= np.trace(rho).real
    val, _ = squashed_entanglement_upper_bound(rho, NA, NB, NE,
                                               n_restarts=3, seed=0,
                                               verbose=True)
    print(f"random 2x2 mixed: E_sq upper bound = {val:.6f} bits")
