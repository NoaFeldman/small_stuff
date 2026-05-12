"""Squashed-entanglement upper bound for the qutrit Werner state.

rho_AB(alpha) = (I + alpha * SWAP) / (d * (d + alpha)),  d = 3,  alpha in [-1, 1].
"""

from __future__ import annotations

import numpy as np

from squashed import squashed_entanglement_upper_bound


def swap_operator(d: int) -> np.ndarray:
    S = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[i * d + j, j * d + i] = 1.0
    return S


def werner_state(alpha: float, d: int = 3) -> np.ndarray:
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [-1, 1].")
    I = np.eye(d * d, dtype=complex)
    return (I + alpha * swap_operator(d)) / (d * (d + alpha))


def werner_squashed_upper_bound(
    alpha: float,
    NE: int,
    *,
    base: float = 2.0,
    n_restarts: int = 8,
    max_iter: int = 2000,
    seed: int | None = 0,
    x0: np.ndarray | None = None,
    return_x: bool = False,
    verbose: bool = False,
):
    d = 3
    rho_AB = werner_state(alpha, d=d)
    return squashed_entanglement_upper_bound(
        rho_AB, NA=d, NB=d, NE=NE,
        base=base, n_restarts=n_restarts, max_iter=max_iter,
        seed=seed, x0=x0, return_x=return_x, verbose=verbose,
    )


def werner_sweep(
    alphas: np.ndarray,
    NE: int,
    *,
    n_restarts: int = 4,
    max_iter: int = 2000,
    seed: int | None = 0,
    verbose: bool = False,
) -> list[dict]:
    """Sweep over `alphas` with warm starts in both directions.

    For each alpha we keep the minimum of the forward-swept and the
    backward-swept solutions (both are valid upper bounds).  At every step
    `n_restarts` random restarts are also tried as a safety net.
    """
    alphas = np.asarray(alphas, dtype=float)
    n = len(alphas)

    fwd_vals = np.empty(n)
    bwd_vals = np.empty(n)

    # Forward sweep.
    x_prev: np.ndarray | None = None
    for i, a in enumerate(alphas):
        if verbose:
            print(f"[fwd {i+1}/{n}] alpha = {a:+.4f}")
        cmi, _, x = werner_squashed_upper_bound(
            float(a), NE,
            n_restarts=n_restarts, max_iter=max_iter, seed=seed,
            x0=x_prev, return_x=True, verbose=verbose,
        )
        fwd_vals[i] = cmi
        x_prev = x

    # Backward sweep.
    x_prev = None
    seed_b = seed + 1 if seed is not None else None
    for i in range(n - 1, -1, -1):
        a = alphas[i]
        if verbose:
            print(f"[bwd {n-i}/{n}] alpha = {a:+.4f}")
        cmi, _, x = werner_squashed_upper_bound(
            float(a), NE,
            n_restarts=n_restarts, max_iter=max_iter, seed=seed_b,
            x0=x_prev, return_x=True, verbose=verbose,
        )
        bwd_vals[i] = cmi
        x_prev = x

    out: list[dict] = []
    for i, a in enumerate(alphas):
        if fwd_vals[i] <= bwd_vals[i]:
            out.append({"alpha": float(a), "NE": NE,
                        "cmi_upper_bound_bits": float(fwd_vals[i]),
                        "direction": "fwd"})
        else:
            out.append({"alpha": float(a), "NE": NE,
                        "cmi_upper_bound_bits": float(bwd_vals[i]),
                        "direction": "bwd"})
    return out
