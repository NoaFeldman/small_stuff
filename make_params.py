"""Generate the parameter grid: one line = one SLURM array task = one NE value."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--NE-list", type=int, nargs="+", default=[3, 6, 9])
    p.add_argument("--out", type=str, default="params.txt")
    args = p.parse_args()

    with open(args.out, "w") as f:
        f.write("\n".join(str(NE) for NE in args.NE_list) + "\n")

    print(f"Wrote {len(args.NE_list)} NE values to {args.out}")
    print(f"  NE values: {args.NE_list}")
    print(f"  SLURM array range: 1-{len(args.NE_list)}")


if __name__ == "__main__":
    main()
