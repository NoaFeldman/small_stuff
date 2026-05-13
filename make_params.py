"""Generate the parameter grid: one line = one SLURM array task = one (NE, alpha)."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--NE-list", type=int, nargs="+", default=[3, 6, 9])
    p.add_argument("--alpha-min", type=float, default=-1.0)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--n-alpha", type=int, default=21)
    p.add_argument("--out", type=str, default="params.txt")
    args = p.parse_args()

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)

    lines = [f"{NE} {a:.10f}" for NE in args.NE_list for a in alphas]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} (NE, alpha) tasks to {args.out}")
    print(f"  NE values:  {args.NE_list}")
    print(f"  alpha grid: {args.n_alpha} points in [{args.alpha_min}, {args.alpha_max}]")
    print(f"  SLURM array range: 1-{len(lines)}")


if __name__ == "__main__":
    main()
