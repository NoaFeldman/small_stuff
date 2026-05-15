"""Generate parameter grid for the squashed entanglement scan.

alpha = -1/3 + 0.01/6 * i  for i in range(-5, 6)   (11 values)
NE in [3, 6, 9, 12, 15]                              (5 values)
=> 55 tasks total.
"""

from __future__ import annotations


def main() -> None:
    NE_list = [3, 6, 9, 12, 15]
    alphas = [-1.0 / 3.0 + 0.01 / 6.0 * i for i in range(-5, 6)]

    lines = [f"{NE} {a:.15f}" for NE in NE_list for a in alphas]
    out = "params_esq.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} (NE, alpha) tasks to {out}")
    print(f"  NE values:  {NE_list}")
    print(f"  alpha grid: {len(alphas)} points around -1/3")
    for i, a in enumerate(alphas):
        print(f"    i={i-5:+d}  alpha={a:+.15f}")
    print(f"  SLURM array range: 1-{len(lines)}")


if __name__ == "__main__":
    main()
