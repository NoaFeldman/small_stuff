"""CLI driver: full alpha sweep for one NE value (one SLURM array task)."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from werner import werner_sweep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--NE", type=int, required=True)
    p.add_argument("--alpha-min", type=float, default=-1.0)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--n-alpha", type=int, default=21)
    p.add_argument("--n-restarts", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--task-id", type=str,
                   default=os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)

    t0 = time.time()
    results = werner_sweep(
        alphas, args.NE,
        n_restarts=args.n_restarts,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - t0

    out = {
        "task_id": args.task_id,
        "NE": args.NE,
        "n_restarts": args.n_restarts,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    json_path = os.path.join(args.out_dir, f"task_{args.task_id}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[task {args.task_id}] NE={args.NE}  {len(alphas)} alphas  total {elapsed:.1f} s")
    for r in results:
        print(f"  alpha={r['alpha']:+.4f}  E_sq <= {r['cmi_upper_bound_bits']:.6f}  ({r['direction']})")


if __name__ == "__main__":
    main()
