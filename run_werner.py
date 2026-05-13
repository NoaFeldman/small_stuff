"""CLI driver: one (NE, alpha) point per SLURM array task."""

from __future__ import annotations

import argparse
import json
import os
import time

from werner import werner_squashed_upper_bound


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--NE", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--n-restarts", type=int, default=256)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--task-id", type=str,
                   default=os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    cmi, _ = werner_squashed_upper_bound(
        args.alpha, args.NE,
        n_restarts=args.n_restarts,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - t0

    out = {
        "task_id": args.task_id,
        "NE": args.NE,
        "alpha": args.alpha,
        "cmi_upper_bound_bits": float(cmi),
        "n_restarts": args.n_restarts,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
    }
    json_path = os.path.join(args.out_dir, f"task_{args.task_id}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[task {args.task_id}] NE={args.NE}  alpha={args.alpha:+.4f}  "
          f"E_sq <= {cmi:.6f}  ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
