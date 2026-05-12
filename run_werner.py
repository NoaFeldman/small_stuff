"""CLI driver: compute squashed-entanglement upper bound for one Werner point.

Intended to be launched as a single SLURM array task.  Writes a small JSON
file with the result so that `collect.py` can aggregate everything later.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from werner import werner_squashed_upper_bound


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--NE", type=int, required=True)
    p.add_argument("--n-restarts", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory where the per-task result JSON is written.")
    p.add_argument("--task-id", type=str, default=os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    p.add_argument("--save-rho", action="store_true",
                   help="Also save the optimal rho_ABE as a .npy file.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    cmi, rho_ABE = werner_squashed_upper_bound(
        alpha=args.alpha,
        NE=args.NE,
        n_restarts=args.n_restarts,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=False,
    )
    elapsed = time.time() - t0

    result = {
        "task_id": args.task_id,
        "alpha": args.alpha,
        "NE": args.NE,
        "n_restarts": args.n_restarts,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "cmi_upper_bound_bits": cmi,
        "elapsed_seconds": elapsed,
    }
    json_path = os.path.join(args.out_dir, f"task_{args.task_id}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    if args.save_rho:
        np.save(os.path.join(args.out_dir, f"task_{args.task_id}_rhoABE.npy"), rho_ABE)

    print(f"[task {args.task_id}] alpha={args.alpha} NE={args.NE} "
          f"E_sq <= {cmi:.6f} bits  ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
