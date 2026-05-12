"""Variational upper bound on the squashed entanglement.

Following the user's convention,

    E_sq(A:B) := min_{rho_ABE :  Tr_E rho_ABE = rho_AB}  I(A:B|E),

where I(A:B|E) is the quantum conditional mutual information (see cmi.py).
(Note: many references include an extra factor 1/2 in the definition; we omit
it here to match the stated definition.)

Strategy
--------
Every extension rho_ABE of rho_AB can be obtained as follows
(Stinespring / Koashi-Imoto):

    1. Take a purification |psi> of rho_AB on AB (x) R, where R is a reference
       system of dimension NR = rank(rho_AB).
    2. Apply a quantum channel  Lambda : R -> E  to the R subsystem.

Any such channel has a Stinespring isometry  V : R -> E (x) E_aux  with
NE_aux <= NR.  The extension is then

    rho_ABE = Tr_{E_aux} [(I_AB (x) V) |psi><psi| (I_AB (x) V^dagger)].

We parametrize V as the orthonormal basis (QR factor) of an unconstrained
complex matrix W of shape (NE * NE_aux,  NR) and minimize the CMI over W
with L-BFGS-B (numerical gradient) and a few random restarts.  Because the
optimization is non-convex, the returned value is an *upper bound* on the
true minimum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from cmi import conditional_mutual_information


def _purify(rho: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, int]:
    """Return a purification of rho as a matrix psi of shape (D, NR).

    The full purification vector |psi> in C^D (x) C^NR is psi.flatten()
    (system index first, reference index second).  NR = rank(rho).
    """
    w, U = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    w = np.clip(w.real, 0.0, None)
    mask = w > tol
    w = w[mask]
    U = U[:, mask]
    if w.size == 0:
        raise ValueError("rho is (numerically) zero.")
    psi = U * np.sqrt(w)  # shape (D, NR);  |psi> = sum_i sqrt(w_i) |u_i>|i>
    return psi, int(w.size)


def squashed_entanglement_upper_bound(
    rho_AB: np.ndarray,
    NA: int,
    NB: int,
    NE: int,
    *,
    base: float = 2.0,
    n_restarts: int = 5,
    max_iter: int = 500,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """Variational upper bound on  min_{rho_ABE} I(A:B|E)  with Tr_E rho_ABE = rho_AB.

    Parameters
    ----------
    rho_AB : (NA*NB, NA*NB) density matrix (AB tensor ordering).
    NA, NB, NE : Hilbert-space dimensions of A, B, E.
    base : log base for the entropy / CMI.
    n_restarts : number of random L-BFGS-B restarts (best result is returned).
    max_iter : max iterations per restart.
    seed : RNG seed.
    verbose : print per-restart objective values.

    Returns
    -------
    best_cmi : the smallest CMI found (upper bound on the squashed entanglement).
    best_rho_ABE : the optimal extension achieving `best_cmi`.
    """
    rng = np.random.default_rng(seed)
    D_AB = NA * NB
    if rho_AB.shape != (D_AB, D_AB):
        raise ValueError(f"rho_AB has shape {rho_AB.shape}, expected ({D_AB}, {D_AB}).")

    psi_mat, NR = _purify(rho_AB)  # psi_mat: (D_AB, NR)

    # Stinespring ancilla dimension; NE_aux = NR is always sufficient.
    NE_aux = NR
    rows = NE * NE_aux  # >= NR  (true whenever NE >= 1)
    if rows < NR:
        raise ValueError("NE * NR must be >= NR; choose NE >= 1.")

    n_complex = rows * NR
    n_params = 2 * n_complex  # real and imaginary parts

    def unpack(x: np.ndarray) -> np.ndarray:
        return (x[:n_complex] + 1j * x[n_complex:]).reshape(rows, NR)

    def build_rho_ABE(W: np.ndarray) -> np.ndarray:
        # Isometry V : R -> E (x) E_aux  from QR of W  (shape (rows, NR)).
        V, _ = np.linalg.qr(W)
        # Apply V to the R index of |psi>:  psi'[ab, j] = sum_r psi[ab, r] V[j, r].
        psi_prime = psi_mat @ V.T            # shape (D_AB, rows)
        Psi = psi_prime.reshape(D_AB, NE, NE_aux)
        # rho_ABE = Tr_{E_aux} |Psi><Psi|
        rho = np.einsum('aek,bfk->aebf', Psi, Psi.conj())
        return rho.reshape(D_AB * NE, D_AB * NE)

    def objective(x: np.ndarray) -> float:
        rho_ABE = build_rho_ABE(unpack(x))
        return conditional_mutual_information(rho_ABE, NA, NB, NE, base=base)

    best_val = np.inf
    best_rho: np.ndarray | None = None
    for k in range(n_restarts):
        x0 = rng.standard_normal(n_params) * 0.5
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter})
        val = float(res.fun)
        if verbose:
            print(f"  restart {k+1}/{n_restarts}: CMI = {val:.6f}  (success={res.success})")
        if val < best_val:
            best_val = val
            best_rho = build_rho_ABE(unpack(res.x))

    assert best_rho is not None
    # Sanity: optimized extension reduces to rho_AB.
    from cmi import partial_trace
    rho_AB_recovered = partial_trace(best_rho, (NA, NB, NE), keep=(0, 1))
    err = np.linalg.norm(rho_AB_recovered - rho_AB)
    if err > 1e-8:
        # Should never trigger; the parametrization preserves rho_AB exactly.
        print(f"warning: ||Tr_E rho_ABE - rho_AB|| = {err:.2e}")

    return best_val, best_rho


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # ----- Test 1: product state rho_A (x) rho_B  =>  E_sq = 0. -----
    NA = NB = 2
    NE = 2
    from cmi import _random_density_matrix
    rho_A = _random_density_matrix(NA, rng=rng)
    rho_B = _random_density_matrix(NB, rng=rng)
    rho_AB_prod = np.kron(rho_A, rho_B)
    val, _ = squashed_entanglement_upper_bound(rho_AB_prod, NA, NB, NE,
                                               n_restarts=3, seed=0)
    print(f"product state  E_sq upper bound = {val:.3e}  (expect ~0)")

    # ----- Test 2: maximally entangled pure state on 2x2. -----
    # For pure rho_AB, every extension is rho_AB (x) sigma_E, so
    # I(A:B|E) = I(A:B) = 2 log NA  (= 2 in bits for NA=2).
    d = 2
    phi = np.zeros((d * d,), dtype=complex)
    for i in range(d):
        phi[i * d + i] = 1.0 / np.sqrt(d)
    rho_AB_pure = np.outer(phi, phi.conj())
    val, _ = squashed_entanglement_upper_bound(rho_AB_pure, d, d, NE,
                                               n_restarts=3, seed=0)
    print(f"max-entangled  E_sq upper bound = {val:.6f}  (expect 2.0)")

    # ----- Test 3: generic mixed rho_AB. -----
    rho_AB = _random_density_matrix(NA * NB, rng=rng)
    val, rho_ABE = squashed_entanglement_upper_bound(rho_AB, NA, NB, NE,
                                                     n_restarts=5, seed=0,
                                                     verbose=True)
    print(f"random  rho_AB  E_sq upper bound = {val:.6f}")
