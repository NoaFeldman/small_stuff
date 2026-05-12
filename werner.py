"""Squashed-entanglement upper bound for the qutrit Werner state.

Werner state on two qutrits (NA = NB = d = 3):

    rho_AB(alpha) = (I + alpha * SWAP) / (d * (d + alpha)),

with alpha in [-1, 1]  (SWAP has eigenvalues +/-1 on the symmetric /
antisymmetric subspaces, so positivity requires |alpha| <= 1).
"""

from __future__ import annotations

import numpy as np

from squashed import squashed_entanglement_upper_bound


def swap_operator(d: int) -> np.ndarray:
    """SWAP on C^d (x) C^d:  SWAP |i>|j> = |j>|i>."""
    S = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[i * d + j, j * d + i] = 1.0
    return S


def werner_state(alpha: float, d: int = 3) -> np.ndarray:
    """Werner state  rho(alpha) = (I + alpha SWAP) / (d (d + alpha))."""
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [-1, 1] for a valid Werner state.")
    I = np.eye(d * d, dtype=complex)
    rho = (I + alpha * swap_operator(d)) / (d * (d + alpha))
    return rho


def werner_squashed_upper_bound(
    alpha: float,
    NE: int,
    *,
    base: float = 2.0,
    n_restarts: int = 8,
    max_iter: int = 800,
    seed: int | None = 0,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """Upper bound on E_sq(A:B) for the qutrit Werner state rho_AB(alpha).

    Parameters
    ----------
    alpha : Werner parameter in [-1, 1].
    NE : Hilbert-space dimension of the squashing system E.
    base, n_restarts, max_iter, seed, verbose : forwarded to
        `squashed_entanglement_upper_bound`.

    Returns
    -------
    cmi_min : the smallest CMI found (upper bound on E_sq).
    rho_ABE : the optimal extension achieving `cmi_min`.
    """
    d = 3
    rho_AB = werner_state(alpha, d=d)
    return squashed_entanglement_upper_bound(
        rho_AB, NA=d, NB=d, NE=NE,
        base=base, n_restarts=n_restarts, max_iter=max_iter,
        seed=seed, verbose=verbose,
    )


if __name__ == "__main__":
    # Werner state is separable iff alpha >= 0  (for any d), and entangled
    # iff alpha < 0.  So we expect the upper bound to be ~0 for alpha >= 0
    # and strictly positive for alpha < 0.
    NE = 3
    alphas = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.5, 1.0]
    print(f"{'alpha':>8} | {'E_sq upper bound (bits)':>25}")
    print("-" * 38)
    for a in alphas:
        val, _ = werner_squashed_upper_bound(a, NE, n_restarts=6, seed=0)
        print(f"{a:>8.3f} | {val:>25.6f}")
