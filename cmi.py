"""Quantum conditional mutual information (CMI).

For a tripartite state rho_ABE on H_A (x) H_B (x) H_E:

    I(A:B|E) = S(rho_AE) + S(rho_BE) - S(rho_E) - S(rho_ABE)

where S is the von Neumann entropy.
"""

from __future__ import annotations

import numpy as np


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0, tol: float = 1e-12) -> float:
    """Von Neumann entropy S(rho) = -tr(rho log rho).

    Computed from the (Hermitian) eigenvalues of rho. Eigenvalues below `tol`
    in absolute value are treated as zero (0 log 0 := 0).
    """
    # Symmetrize to remove tiny non-Hermitian numerical drift.
    rho_h = 0.5 * (rho + rho.conj().T)
    eigvals = np.linalg.eigvalsh(rho_h)
    eigvals = np.clip(eigvals.real, 0.0, None)
    eigvals = eigvals[eigvals > tol]
    if eigvals.size == 0:
        return 0.0
    log = np.log(eigvals) / np.log(base)
    return float(-np.sum(eigvals * log))


def partial_trace(rho: np.ndarray, dims: tuple[int, ...], keep: tuple[int, ...]) -> np.ndarray:
    """Partial trace of `rho` over subsystems not in `keep`.

    Parameters
    ----------
    rho : (D, D) array with D = prod(dims).
    dims : Hilbert-space dimensions of each subsystem, in tensor-product order.
    keep : indices of subsystems to keep (0-based, referring to `dims`).
    """
    dims = tuple(int(d) for d in dims)
    n = len(dims)
    D = int(np.prod(dims))
    if rho.shape != (D, D):
        raise ValueError(f"rho has shape {rho.shape}, expected ({D}, {D}).")

    keep = tuple(sorted(set(int(k) for k in keep)))
    if any(k < 0 or k >= n for k in keep):
        raise ValueError(f"`keep` indices {keep} out of range for {n} subsystems.")
    trace_over = tuple(i for i in range(n) if i not in keep)

    # Reshape rho into a rank-2n tensor with row and column indices per subsystem.
    tensor = rho.reshape(dims + dims)

    # Contract each traced subsystem: its row axis with its column axis.
    # einsum letters: rows use 'a'..., cols use 'A'..., matched pairs for traced ones.
    if 2 * n > 26:
        raise ValueError("Too many subsystems for the einsum-letter scheme (max 13).")
    row_letters = [chr(ord('a') + i) for i in range(n)]
    col_letters = [chr(ord('A') + i) for i in range(n)]
    for i in trace_over:
        col_letters[i] = row_letters[i]  # contract row==col on traced subsystems

    in_subs = ''.join(row_letters) + ''.join(col_letters)
    out_subs = ''.join(row_letters[i] for i in keep) + ''.join(col_letters[i] for i in keep)
    reduced = np.einsum(f"{in_subs}->{out_subs}", tensor)

    dkeep = int(np.prod([dims[i] for i in keep])) if keep else 1
    return reduced.reshape(dkeep, dkeep)


def conditional_mutual_information(
    rho_ABE: np.ndarray,
    NA: int,
    NB: int,
    NE: int,
    base: float = 2.0,
) -> float:
    """Quantum CMI I(A:B|E) for rho_ABE on H_A (x) H_B (x) H_E.

    `rho_ABE` must be a (NA*NB*NE, NA*NB*NE) density matrix with subsystems
    ordered A, B, E in the tensor product.
    """
    D = NA * NB * NE
    if rho_ABE.shape != (D, D):
        raise ValueError(f"rho_ABE has shape {rho_ABE.shape}, expected ({D}, {D}).")

    dims = (NA, NB, NE)
    rho_AE = partial_trace(rho_ABE, dims, keep=(0, 2))
    rho_BE = partial_trace(rho_ABE, dims, keep=(1, 2))
    rho_E = partial_trace(rho_ABE, dims, keep=(2,))

    S_ABE = von_neumann_entropy(rho_ABE, base=base)
    S_AE = von_neumann_entropy(rho_AE, base=base)
    S_BE = von_neumann_entropy(rho_BE, base=base)
    S_E = von_neumann_entropy(rho_E, base=base)

    return S_AE + S_BE - S_E - S_ABE


def _random_density_matrix(d: int, rank: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random density matrix of dimension d via X X^dagger / tr(...)."""
    if rng is None:
        rng = np.random.default_rng()
    if rank is None:
        rank = d
    X = rng.standard_normal((d, rank)) + 1j * rng.standard_normal((d, rank))
    rho = X @ X.conj().T
    rho /= np.trace(rho).real
    return rho


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    NA, NB, NE = 2, 2, 2
    D = NA * NB * NE

    # 1) Product state rho_A (x) rho_B (x) rho_E  => I(A:B|E) = 0.
    rho_A = _random_density_matrix(NA, rng=rng)
    rho_B = _random_density_matrix(NB, rng=rng)
    rho_E = _random_density_matrix(NE, rng=rng)
    rho_prod = np.kron(np.kron(rho_A, rho_B), rho_E)
    print(f"I(A:B|E) on product state  = {conditional_mutual_information(rho_prod, NA, NB, NE):.3e}")

    # 2) Generic random state.
    rho = _random_density_matrix(D, rng=rng)
    print(f"I(A:B|E) on random state   = {conditional_mutual_information(rho, NA, NB, NE):.6f}")
